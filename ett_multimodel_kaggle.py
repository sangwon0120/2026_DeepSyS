"""
Leakage-safe ETTh1 multivariate-to-univariate forecasting pipeline.

Kaggle notebook setup:
    # PyTorch is normally preinstalled on Kaggle.
    # !pip install lightgbm

Recommended local/Kaggle run:
    !python ett_multimodel_kaggle.py --patch-seeds 42,2024,2026 --output multimodel_output/submit.csv

Legacy direct multi-model comparison:
    !python ett_multimodel_kaggle.py --legacy-multimodel --patch-seeds 42,2024,2026

Fast leakage and submission-schema check without model training:
    !python ett_multimodel_kaggle.py --validate-only --output multimodel_output/submit_validate_only.csv

This file intentionally keeps test inference separate from supervised target
construction. Test-period OT observations may be used only when they occur
strictly before each submission timestamp.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    TORCH_AVAILABLE = False


# Core competition settings requested by the project specification.
USE_MULTIVARIATE_INPUT = True
FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_COL = "OT"
HORIZON = 96
LEGACY_THREEWAY_ANCHOR_RECIPE = "threeway_prevweek_4block_shrink75"
DEFAULT_THREEWAY_ANCHOR_RECIPE = "threeway_week2blend_12block_shrink50"

RAW_TIME_FEATURE_COLS = [
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
]
CYCLIC_TIME_FEATURE_COLS = [
    "sin_hour",
    "cos_hour",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
]
TIME_FEATURE_COLS = [*RAW_TIME_FEATURE_COLS, *CYCLIC_TIME_FEATURE_COLS]
TEST_START = datetime(2018, 2, 1, 0, 0, 0)
LAST_ALLOWED_TARGET_START = datetime(2018, 1, 28, 0, 0, 0)
LAST_ALLOWED_TARGET_END = datetime(2018, 1, 31, 23, 0, 0)
EXPECTED_LAST_TEST_ORIGIN = datetime(2018, 6, 22, 0, 0, 0)

PATCHTST_LOOKBACKS = (512, 336)
DLINEAR_LOOKBACK = 336
NBEATS_LOOKBACK = 336
NHITS_LOOKBACK = 336
MAX_LOOKBACK = max(*PATCHTST_LOOKBACKS, DLINEAR_LOOKBACK, NBEATS_LOOKBACK, NHITS_LOOKBACK)

FIXED_WEIGHTS = {
    "PatchTST": 0.40,
    "DLinear/NLinear": 0.25,
    "N-BEATS": 0.15,
    "N-HiTS": 0.10,
    "Seasonal Naive": 0.10,
}
FIXED_WEIGHTS_WITH_GBM = {
    "PatchTST": 0.35,
    "DLinear/NLinear": 0.20,
    "N-BEATS": 0.10,
    "N-HiTS": 0.10,
    "Seasonal Naive": 0.10,
    "GBM optional": 0.15,
}
GBM_LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 96, 168, 336)
GBM_ROLL_WINDOWS = (6, 12, 24, 48, 96, 168, 336)
TREE_ANCHOR_LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25)
TREE_ANCHOR_DEFAULT_LAMBDA = 0.25
TREE_ANCHOR_TRAIN_STEP_HOURS = 6
TREE_ANCHOR_PREDICT_CHUNK_ORIGINS = 128


@dataclass
class RawData:
    dates: list[datetime]
    date_to_idx: dict[datetime, int]
    numeric_values: np.ndarray
    sample_ids: list[str]
    sample_datetimes: list[datetime]
    sample_columns: list[str]
    sample_id_col: str


@dataclass
class SplitData:
    test_start_idx: int
    train_origins: np.ndarray
    val_origins: np.ndarray
    test_origins: np.ndarray
    scaler_fit_end_idx: int


@dataclass
class PreparedData:
    raw: RawData
    split: SplitData
    feature_names: list[str]
    features_raw: np.ndarray
    features_scaled: np.ndarray
    target_raw: np.ndarray
    target_scaled: np.ndarray
    target_mean: float
    target_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe ETTh1 direct forecasting ensemble.")
    parser.add_argument("--data", type=Path, default=Path("ETTh1.csv"))
    parser.add_argument("--sample", type=Path, default=Path("csvFiles/sample_submit.csv"))
    parser.add_argument("--output", type=Path, default=Path("multimodel_output/submit.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("multimodel_output/run_report.json"))
    parser.add_argument("--univariate-input", action="store_true", help="Use OT only instead of raw and time features.")
    parser.add_argument("--patch-seeds", default="42", help="Comma-separated PatchTST seeds, for example 42,2024,2026.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--skip-gbm", action="store_true", help="Do not attempt the optional LightGBM model.")
    parser.add_argument(
        "--legacy-multimodel",
        action="store_true",
        help="Run the previous absolute-value direct multi-model comparison instead of the PatchTST residual pipeline.",
    )
    parser.add_argument(
        "--include-raw-calendar",
        action="store_true",
        help="Add raw hour/dayofweek/month/dayofyear features. The default keeps cyclic time features only.",
    )
    parser.add_argument(
        "--patch-lookbacks",
        default="336",
        help="Comma-separated PatchTST residual lookbacks. Use 336,512 to validate a blended candidate.",
    )
    parser.add_argument(
        "--residual-lambdas",
        default="0,0.05,0.07,0.08,0.09,0.10,0.11,0.12,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00",
        help="Comma-separated anchor + lambda * PatchTST residual strengths.",
    )
    parser.add_argument(
        "--block-lambda-search",
        action="store_true",
        help="Also search smooth 24-hour block residual lambdas for T0-T23, ..., T72-T95.",
    )
    parser.add_argument("--block-lambda-values", default="0,0.05,0.10,0.15")
    parser.add_argument("--block-lambda-max-adjacent-diff", type=float, default=0.05)
    parser.add_argument(
        "--horizon-decay-lambda-search",
        action="store_true",
        help="Search conservative 24-hour block residual lambdas that shrink toward longer horizons.",
    )
    parser.add_argument(
        "--horizon-decay-lambda-candidates",
        default="0.12,0.10,0.08,0.06;0.10,0.10,0.08,0.06;0.10,0.08,0.08,0.06;0.10,0.10,0.10,0.08;0.12,0.12,0.10,0.08;0.08,0.08,0.06,0.05",
        help="Semicolon-separated T0-T23,T24-T47,T48-T71,T72-T95 lambda candidates.",
    )
    parser.add_argument(
        "--anchor",
        choices=("threeway", "seasonal-naive"),
        default="threeway",
        help="Leakage-safe anchor forecast. The default fits a three-way seasonal block blend.",
    )
    parser.add_argument(
        "--anchor-analysis-only",
        action="store_true",
        help="Compare conservative anchor candidates without training PatchTST and write the provisional best submission.",
    )
    parser.add_argument(
        "--anchor-rolling-backtest",
        action="store_true",
        help="Run rolling backtests for conservative anchor promotion candidates without training PatchTST.",
    )
    parser.add_argument(
        "--tree-anchor-rolling-backtest",
        action="store_true",
        help="Run leakage-safe LightGBM residual rolling diagnostics for the promoted heuristic anchor.",
    )
    parser.add_argument(
        "--tree-anchor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a LightGBM residual correction as the PatchTST base anchor after rolling promotion.",
    )
    parser.add_argument("--tree-anchor-lambda", type=float, default=TREE_ANCHOR_DEFAULT_LAMBDA)
    parser.add_argument("--tree-anchor-train-step-hours", type=int, default=TREE_ANCHOR_TRAIN_STEP_HOURS)
    parser.add_argument("--tree-anchor-estimators", type=int, default=600)
    parser.add_argument(
        "--final-tree-blend-search",
        action="store_true",
        help="After PatchTST residual selection, search a small LightGBM residual correction as a final blend layer.",
    )
    parser.add_argument(
        "--final-tree-blend-weights",
        default="-0.05,-0.03,0,0.03,0.05,0.07,0.10,0.15",
        help="Comma-separated weights for final_prediction + weight * LightGBM residual.",
    )
    parser.add_argument(
        "--final-tree-blend-min-positive-segments",
        type=int,
        default=3,
        help="Minimum validation segments where the final LightGBM blend must beat the selected PatchTST prediction.",
    )
    parser.add_argument(
        "--patch-residual-rolling-backtest",
        action="store_true",
        help="Run nested rolling diagnostics for PatchTST residual correction without writing a Kaggle submission.",
    )
    parser.add_argument("--rolling-patch-seed", type=int, default=42)
    parser.add_argument("--rolling-pretrain-epochs", type=int, default=3)
    parser.add_argument("--rolling-finetune-epochs", type=int, default=8)
    parser.add_argument("--rolling-patience", type=int, default=3)
    parser.add_argument("--patch-learning-rate", type=float, default=3e-4)
    parser.add_argument("--patch-weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-dropout", type=float, default=0.2)
    parser.add_argument(
        "--patch-ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Track an exponential moving average of PatchTST weights and evaluate it as a residual candidate.",
    )
    parser.add_argument("--patch-ema-decay", type=float, default=0.995)
    parser.add_argument(
        "--patch-seed-candidate-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate PatchTST seed-subset candidates on validation and refit only the selected seeds.",
    )
    parser.add_argument(
        "--selection-mse-tolerance",
        type=float,
        default=0.005,
        help="Among eligible residual candidates within this relative MSE tolerance, prefer the more stable candidate.",
    )
    parser.add_argument(
        "--selection-min-segment-improvement-floor",
        type=float,
        default=0.02,
        help="Preferred candidates should improve every validation segment by at least this MSE margin when available.",
    )
    parser.add_argument(
        "--patch-window-revin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply RevIN-style per-window normalization to numeric PatchTST inputs while preserving level context.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=8, help="All-hour PatchTST residual pretraining epochs.")
    parser.add_argument("--finetune-epochs", type=int, default=20, help="Midnight-origin PatchTST residual finetuning epochs.")
    parser.add_argument(
        "--validation-segments",
        type=int,
        default=3,
        help="Chronological validation segments used to reject unstable residual corrections.",
    )
    parser.add_argument(
        "--min-positive-validation-segments",
        type=int,
        default=3,
        help="Minimum segments where a residual correction must beat its anchor.",
    )
    parser.add_argument(
        "--no-final-refit",
        action="store_true",
        help="Skip the pre-test full-data refit. Useful only for debugging; final submissions should keep refit enabled.",
    )
    parser.add_argument("--quick", action="store_true", help="Train one epoch per neural model for a pipeline smoke run.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run leakage checks and write a seasonal-naive schema-check submission without importing PyTorch models.",
    )
    return parser.parse_args()


def stage(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def parse_datetime(value: str) -> datetime:
    value = value.strip()
    if len(value) == 10:
        return datetime.strptime(value, "%Y-%m-%d")
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def find_file(path: Path, filename: str) -> Path:
    if path.exists():
        return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"{filename} not found at {path} or below /kaggle/input.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def hourly_range(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(hours=1)


def causal_forward_fill(values: np.ndarray) -> tuple[np.ndarray, int]:
    out = values.copy()
    missing_before = int(np.isnan(out).sum())
    for col_idx in range(out.shape[1]):
        if np.isnan(out[0, col_idx]):
            raise ValueError(
                f"Column {FEATURE_COLS[col_idx]} starts with a missing value. "
                "A causal forward fill cannot repair the first observation."
            )
        for row_idx in range(1, out.shape[0]):
            if np.isnan(out[row_idx, col_idx]):
                out[row_idx, col_idx] = out[row_idx - 1, col_idx]
    assert not np.isnan(out).any()
    return out.astype(np.float32), missing_before


def load_data(data_path: Path, sample_path: Path) -> RawData:
    stage("1. Load and clean hourly data")
    data_path = find_file(data_path, "ETTh1.csv")
    sample_path = find_file(sample_path, "sample_submit.csv")

    rows_by_date: dict[datetime, list[float]] = {}
    with data_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing_cols = [col for col in ["date", *FEATURE_COLS] if col not in columns]
        if missing_cols:
            raise ValueError(f"Missing ETTh1 columns: {missing_cols}")
        for row_number, row in enumerate(reader, start=2):
            try:
                dt = parse_datetime(row["date"])
                values = [float(row[col]) if row[col].strip() else float("nan") for col in FEATURE_COLS]
            except Exception as exc:
                raise ValueError(f"Failed to parse {data_path}:{row_number}: {exc}") from exc
            rows_by_date[dt] = values

    sorted_dates = sorted(rows_by_date)
    dates = list(hourly_range(sorted_dates[0], sorted_dates[-1]))
    numeric_values = np.asarray(
        [rows_by_date.get(dt, [float("nan")] * len(FEATURE_COLS)) for dt in dates],
        dtype=np.float32,
    )
    numeric_values, filled_count = causal_forward_fill(numeric_values)

    with sample_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        sample_columns = reader.fieldnames or []
        if "timestamp" in sample_columns:
            sample_id_col = "timestamp"
        elif "ID" in sample_columns:
            sample_id_col = "ID"
        else:
            raise ValueError("sample_submit.csv must contain either timestamp or ID.")
        expected_columns = [sample_id_col, *[f"T{i}" for i in range(HORIZON)]]
        if sample_columns != expected_columns:
            raise ValueError(f"Unexpected submission columns. Expected {expected_columns}, found {sample_columns}.")
        sample_rows = list(reader)

    sample_ids = [row[sample_id_col] for row in sample_rows]
    sample_datetimes = [parse_datetime(value) for value in sample_ids]
    date_to_idx = {dt: idx for idx, dt in enumerate(dates)}

    print(f"data path: {data_path}")
    print(f"data rows after hourly reindex: {len(dates):,}")
    print(f"data range: {dates[0]} ~ {dates[-1]}")
    print(f"missing numeric cells repaired with causal forward fill: {filled_count}")
    print(f"sample path: {sample_path}")
    print(f"sample rows: {len(sample_ids):,}")
    print(f"sample schema: {sample_columns[:4]} ... {sample_columns[-3:]}")
    if sample_id_col != "timestamp":
        print("note: the provided sample uses ID as its timestamp column; submit.csv preserves that Kaggle schema.")

    assert len(dates) == len(date_to_idx)
    assert all((dates[idx + 1] - dates[idx]) == timedelta(hours=1) for idx in range(len(dates) - 1))
    assert sample_datetimes[0] == TEST_START
    assert sample_datetimes[-1] == EXPECTED_LAST_TEST_ORIGIN
    return RawData(
        dates=dates,
        date_to_idx=date_to_idx,
        numeric_values=numeric_values,
        sample_ids=sample_ids,
        sample_datetimes=sample_datetimes,
        sample_columns=sample_columns,
        sample_id_col=sample_id_col,
    )


def datetime_features(dt: datetime) -> list[float]:
    hour = dt.hour
    dayofweek = dt.weekday()
    month = dt.month
    dayofyear = dt.timetuple().tm_yday
    return [
        float(hour),
        float(dayofweek),
        float(month),
        float(dayofyear),
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
        math.sin(2.0 * math.pi * dayofweek / 7.0),
        math.cos(2.0 * math.pi * dayofweek / 7.0),
        math.sin(2.0 * math.pi * (dayofyear - 1) / 366.0),
        math.cos(2.0 * math.pi * (dayofyear - 1) / 366.0),
    ]


def make_feature_matrix(
    raw: RawData,
    use_multivariate_input: bool,
    include_raw_calendar: bool,
) -> tuple[np.ndarray, list[str]]:
    if not use_multivariate_input:
        target_idx = FEATURE_COLS.index(TARGET_COL)
        return raw.numeric_values[:, [target_idx]].copy(), [TARGET_COL]
    selected_time_cols = TIME_FEATURE_COLS if include_raw_calendar else CYCLIC_TIME_FEATURE_COLS
    selected_time_indices = [TIME_FEATURE_COLS.index(col) for col in selected_time_cols]
    time_values = np.asarray([datetime_features(dt) for dt in raw.dates], dtype=np.float32)[:, selected_time_indices]
    features = np.concatenate([raw.numeric_values, time_values], axis=1)
    return features.astype(np.float32), [*FEATURE_COLS, *selected_time_cols]


def make_midnight_origins(raw: RawData, first_idx: int, last_idx: int) -> np.ndarray:
    origins = [idx for idx in range(first_idx, last_idx + 1) if raw.dates[idx].hour == 0]
    return np.asarray(origins, dtype=np.int64)


def make_all_hour_origins(first_idx: int, last_idx: int) -> np.ndarray:
    return np.arange(first_idx, last_idx + 1, dtype=np.int64)


def format_target_range(raw: RawData, origins: np.ndarray) -> str:
    return (
        f"y_start={raw.dates[int(origins[0])]} ~ {raw.dates[int(origins[-1])]}, "
        f"y_end={raw.dates[int(origins[0]) + HORIZON - 1]} ~ {raw.dates[int(origins[-1]) + HORIZON - 1]}"
    )


def build_splits(raw: RawData, validation_fraction: float) -> SplitData:
    stage("2. Build leakage-safe train, validation, and test origins")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    if TEST_START not in raw.date_to_idx:
        raise ValueError(f"Missing test boundary row: {TEST_START}")

    test_start_idx = raw.date_to_idx[TEST_START]
    max_allowed_origin_idx = test_start_idx - HORIZON
    allowed_origins = make_midnight_origins(raw, MAX_LOOKBACK, max_allowed_origin_idx)
    val_count = max(1, int(math.ceil(len(allowed_origins) * validation_fraction)))
    val_origins = allowed_origins[-val_count:]

    # Purge overlapping target labels: the last train y must end before the
    # first validation y starts.
    first_val_origin = int(val_origins[0])
    train_origins = allowed_origins[allowed_origins + HORIZON <= first_val_origin]
    test_origins = np.asarray([raw.date_to_idx[dt] for dt in raw.sample_datetimes], dtype=np.int64)

    print(f"allowed supervised origin count: {len(allowed_origins):,}")
    print(f"train samples: {len(train_origins):,}; {format_target_range(raw, train_origins)}")
    print(f"validation samples: {len(val_origins):,}; {format_target_range(raw, val_origins)}")
    print(f"last allowed supervised origin: {raw.dates[int(allowed_origins[-1])]}")
    print(f"last allowed supervised y_end: {raw.dates[int(allowed_origins[-1]) + HORIZON - 1]}")
    print(f"scaler fit row range: {raw.dates[0]} ~ {raw.dates[first_val_origin - 1]}")
    print(f"test origin range: {raw.dates[int(test_origins[0])]} ~ {raw.dates[int(test_origins[-1])]}")

    assert len(train_origins) > 0 and len(val_origins) > 0
    assert raw.dates[int(allowed_origins[-1])] == LAST_ALLOWED_TARGET_START
    assert raw.dates[int(allowed_origins[-1]) + HORIZON - 1] == LAST_ALLOWED_TARGET_END
    assert all(raw.dates[int(idx)].hour == 0 for idx in train_origins)
    assert all(raw.dates[int(idx)].hour == 0 for idx in val_origins)
    assert all(raw.dates[int(idx)].hour == 0 for idx in test_origins)
    assert int(train_origins[-1]) + HORIZON <= int(val_origins[0])
    assert raw.dates[int(val_origins[-1]) + HORIZON - 1] <= LAST_ALLOWED_TARGET_END

    for timestamp, origin_idx in zip(raw.sample_datetimes, test_origins):
        actual_input_end = raw.dates[int(origin_idx) - 1]
        expected_input_end = timestamp - timedelta(hours=1)
        assert actual_input_end == expected_input_end
        assert actual_input_end.date() == (timestamp - timedelta(days=1)).date()
        assert actual_input_end.hour == 23
    print("test input assertion: every X window ends exactly at timestamp - 1 hour (previous day 23:00).")

    return SplitData(
        test_start_idx=test_start_idx,
        train_origins=train_origins,
        val_origins=val_origins,
        test_origins=test_origins,
        scaler_fit_end_idx=first_val_origin,
    )


def prepare_data(
    raw: RawData,
    split: SplitData,
    use_multivariate_input: bool,
    include_raw_calendar: bool = False,
    scaler_fit_end_idx: int | None = None,
) -> PreparedData:
    stage("3. Fit scaler on the train period only")
    scaler_fit_end_idx = scaler_fit_end_idx or split.scaler_fit_end_idx
    features_raw, feature_names = make_feature_matrix(raw, use_multivariate_input, include_raw_calendar)
    train_rows = features_raw[:scaler_fit_end_idx]
    feature_mean = train_rows.mean(axis=0)
    feature_std = train_rows.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    features_scaled = ((features_raw - feature_mean) / feature_std).astype(np.float32)

    target_idx = FEATURE_COLS.index(TARGET_COL)
    target_raw = raw.numeric_values[:, target_idx].astype(np.float32)
    target_mean = float(target_raw[:scaler_fit_end_idx].mean())
    target_std = float(target_raw[:scaler_fit_end_idx].std())
    if target_std < 1e-6:
        target_std = 1.0
    target_scaled = ((target_raw - target_mean) / target_std).astype(np.float32)

    print(f"input mode: {'multivariate + time features' if use_multivariate_input else 'univariate OT only'}")
    print(f"model feature count: {len(feature_names)}")
    print(f"model features: {feature_names}")
    print(f"target scaler: mean={target_mean:.6f}, std={target_std:.6f}")
    print(f"scaler fit end exclusive: {raw.dates[scaler_fit_end_idx]}")
    assert scaler_fit_end_idx <= split.test_start_idx
    return PreparedData(
        raw=raw,
        split=split,
        feature_names=feature_names,
        features_raw=features_raw,
        features_scaled=features_scaled,
        target_raw=target_raw,
        target_scaled=target_scaled,
        target_mean=target_mean,
        target_std=target_std,
    )


def make_supervised_targets(target_values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Build labels only for pre-test train/validation origins, never for test origins."""
    return np.stack([target_values[int(origin) : int(origin) + HORIZON] for origin in origins]).astype(np.float32)


def make_seasonal_naive_predictions(target_values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    predictions = []
    for origin in origins:
        last_day = target_values[int(origin) - 24 : int(origin)]
        assert last_day.shape == (24,)
        predictions.append(np.tile(last_day, HORIZON // 24))
    return np.asarray(predictions, dtype=np.float32)


def make_last_value_predictions(target_values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.repeat(target_values[int(origin) - 1], HORIZON) for origin in origins],
        dtype=np.float32,
    )


def make_previous_window_predictions(target_values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    return np.asarray(
        [target_values[int(origin) - HORIZON : int(origin)] for origin in origins],
        dtype=np.float32,
    )


def make_previous_week_predictions(target_values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    week = 24 * 7
    return np.asarray(
        [target_values[int(origin) - week : int(origin) - week + HORIZON] for origin in origins],
        dtype=np.float32,
    )


def make_previous_two_week_predictions(target_values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    two_weeks = 24 * 14
    return np.asarray(
        [target_values[int(origin) - two_weeks : int(origin) - two_weeks + HORIZON] for origin in origins],
        dtype=np.float32,
    )


def make_level_adjusted_week_predictions(
    target_values: np.ndarray,
    origins: np.ndarray,
    clip: float,
) -> np.ndarray:
    """Shift last week's profile by a clipped recent 24-hour level change."""
    week = 24 * 7
    predictions = []
    for origin in origins:
        origin = int(origin)
        recent_day = target_values[origin - 24 : origin]
        previous_week_day = target_values[origin - week - 24 : origin - week]
        adjustment = float(np.mean(recent_day - previous_week_day))
        predictions.append(
            target_values[origin - week : origin - week + HORIZON] + np.clip(adjustment, -clip, clip)
        )
    return np.asarray(predictions, dtype=np.float32)


def fit_level_adjustment_clip(target_values: np.ndarray, origins: np.ndarray, quantile: float = 0.75) -> float:
    week = 24 * 7
    adjustments = []
    for origin in origins:
        origin = int(origin)
        recent_day = target_values[origin - 24 : origin]
        previous_week_day = target_values[origin - week - 24 : origin - week]
        adjustments.append(abs(float(np.mean(recent_day - previous_week_day))))
    return max(float(np.quantile(adjustments, quantile)), 1e-6)


def fit_convex_blend_weight(y_true: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray) -> float:
    difference = prediction_a - prediction_b
    denominator = float(np.sum(difference * difference))
    if denominator <= 1e-12:
        return 1.0
    weight = float(np.sum((y_true - prediction_b) * difference) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def make_seasonal_member_predictions(target_values: np.ndarray, origins: np.ndarray, predictor: dict) -> np.ndarray:
    seasonal_member = predictor["seasonal_member"]
    if seasonal_member == "previous_week":
        return make_previous_week_predictions(target_values, origins)
    if seasonal_member == "level_adjusted_week":
        return make_level_adjusted_week_predictions(target_values, origins, clip=float(predictor["level_clip"]))
    if seasonal_member == "week_two_week_blend":
        week_weight = float(predictor["week_weight"])
        return (
            week_weight * make_previous_week_predictions(target_values, origins)
            + (1.0 - week_weight) * make_previous_two_week_predictions(target_values, origins)
        ).astype(np.float32)
    raise ValueError(f"unknown seasonal anchor member: {seasonal_member}")


def fit_threeway_block_weights(
    target_values: np.ndarray,
    origins: np.ndarray,
    block_size: int,
    seasonal_predictions: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a convex blend from pre-test labels only."""
    true = make_supervised_targets(target_values, origins)
    seasonal_predictions = seasonal_predictions if seasonal_predictions is not None else make_previous_week_predictions(
        target_values,
        origins,
    )
    members = np.stack(
        [
            make_last_value_predictions(target_values, origins),
            make_previous_window_predictions(target_values, origins),
            seasonal_predictions,
        ],
        axis=-1,
    )
    weights = []
    grid = np.linspace(0.0, 1.0, 21)
    for start in range(0, HORIZON, block_size):
        end = min(start + block_size, HORIZON)
        block_true = true[:, start:end]
        block_members = members[:, start:end]
        best_mse = float("inf")
        best_weights = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        for last_weight in grid:
            for previous_weight in grid:
                week_weight = 1.0 - last_weight - previous_weight
                if week_weight < -1e-8:
                    continue
                candidate_weights = np.asarray([last_weight, previous_weight, week_weight], dtype=np.float32)
                prediction = np.sum(block_members * candidate_weights, axis=-1)
                candidate_mse = metric(block_true, prediction)["mse"]
                if candidate_mse < best_mse:
                    best_mse = candidate_mse
                    best_weights = candidate_weights
        weights.append(np.repeat(best_weights[None, :], end - start, axis=0))
    return np.concatenate(weights, axis=0)


def fit_threeway_anchor_candidate(
    target_values: np.ndarray,
    origins: np.ndarray,
    recipe: dict,
) -> dict:
    predictor = dict(recipe)
    seasonal_member = predictor["seasonal_member"]
    if seasonal_member == "level_adjusted_week":
        predictor["level_clip"] = fit_level_adjustment_clip(target_values, origins)
    elif seasonal_member == "week_two_week_blend":
        true = make_supervised_targets(target_values, origins)
        predictor["week_weight"] = fit_convex_blend_weight(
            true,
            make_previous_week_predictions(target_values, origins),
            make_previous_two_week_predictions(target_values, origins),
        )
    seasonal_predictions = make_seasonal_member_predictions(target_values, origins, predictor)
    coarse_weights = fit_threeway_block_weights(target_values, origins, block_size=24, seasonal_predictions=seasonal_predictions)
    fine_block_size = int(predictor["fine_block_size"])
    if fine_block_size == 24:
        weights = coarse_weights
    else:
        fine_weights = fit_threeway_block_weights(
            target_values,
            origins,
            block_size=fine_block_size,
            seasonal_predictions=seasonal_predictions,
        )
        fine_amount = float(predictor["fine_amount"])
        weights = fine_amount * fine_weights + (1.0 - fine_amount) * coarse_weights
    predictor["kind"] = "threeway"
    predictor["weights"] = weights.astype(np.float32)
    return predictor


def anchor_candidate_recipes() -> list[dict]:
    """Small, deliberately constrained candidate set for rolling validation."""
    return [
        {
            "name": "threeway_prevweek_24block",
            "recipe": "threeway_prevweek_24block",
            "seasonal_member": "previous_week",
            "fine_block_size": 24,
            "fine_amount": 0.0,
            "complexity": 1,
        },
        {
            "name": "threeway_prevweek_12block_shrink50",
            "recipe": "threeway_prevweek_12block_shrink50",
            "seasonal_member": "previous_week",
            "fine_block_size": 12,
            "fine_amount": 0.50,
            "complexity": 2,
        },
        {
            "name": "threeway_prevweek_4block_shrink50",
            "recipe": "threeway_prevweek_4block_shrink50",
            "seasonal_member": "previous_week",
            "fine_block_size": 4,
            "fine_amount": 0.50,
            "complexity": 3,
        },
        {
            "name": "threeway_prevweek_4block_shrink75",
            "recipe": "threeway_prevweek_4block_shrink75",
            "seasonal_member": "previous_week",
            "fine_block_size": 4,
            "fine_amount": 0.75,
            "complexity": 4,
        },
        {
            "name": "threeway_levelweek_24block",
            "recipe": "threeway_levelweek_24block",
            "seasonal_member": "level_adjusted_week",
            "fine_block_size": 24,
            "fine_amount": 0.0,
            "complexity": 2,
        },
        {
            "name": "threeway_levelweek_12block_shrink50",
            "recipe": "threeway_levelweek_12block_shrink50",
            "seasonal_member": "level_adjusted_week",
            "fine_block_size": 12,
            "fine_amount": 0.50,
            "complexity": 3,
        },
        {
            "name": "threeway_week2blend_24block",
            "recipe": "threeway_week2blend_24block",
            "seasonal_member": "week_two_week_blend",
            "fine_block_size": 24,
            "fine_amount": 0.0,
            "complexity": 2,
        },
        {
            "name": "threeway_week2blend_12block_shrink50",
            "recipe": "threeway_week2blend_12block_shrink50",
            "seasonal_member": "week_two_week_blend",
            "fine_block_size": 12,
            "fine_amount": 0.50,
            "complexity": 3,
        },
    ]


def fit_anchor_candidate(target_values: np.ndarray, origins: np.ndarray, recipe_name: str) -> dict:
    recipe = next((item for item in anchor_candidate_recipes() if item["recipe"] == recipe_name), None)
    if recipe is None:
        raise ValueError(f"unknown anchor recipe: {recipe_name}")
    return fit_threeway_anchor_candidate(target_values, origins, recipe)


def fit_anchor_candidates(target_values: np.ndarray, origins: np.ndarray) -> list[dict]:
    return [fit_threeway_anchor_candidate(target_values, origins, recipe) for recipe in anchor_candidate_recipes()]


def evaluate_anchor_candidates(
    target_values: np.ndarray,
    origins: np.ndarray,
    candidates: list[dict],
) -> list[dict]:
    true = make_supervised_targets(target_values, origins)
    rows = []
    for predictor in candidates:
        rows.append(
            {
                "name": predictor["name"],
                "recipe": predictor["recipe"],
                "complexity": int(predictor["complexity"]),
                **metric(true, make_anchor_predictions(target_values, origins, predictor)),
            }
        )
    return sorted(rows, key=lambda row: (row["mse"], row["complexity"], row["name"]))


def fit_anchor_predictor(target_values: np.ndarray, origins: np.ndarray, anchor: str) -> dict:
    if anchor == "seasonal-naive":
        return {"name": "seasonal_naive_24h", "kind": anchor}
    if anchor != "threeway":
        raise ValueError(f"unknown anchor: {anchor}")
    if len(origins) == 0 or int(np.min(origins)) < 24 * 14:
        raise ValueError("threeway anchor requires training origins with at least two weeks of history.")
    return fit_anchor_candidate(target_values, origins, DEFAULT_THREEWAY_ANCHOR_RECIPE)


def make_anchor_predictions(target_values: np.ndarray, origins: np.ndarray, predictor: dict) -> np.ndarray:
    """Predict from observations strictly before each origin."""
    if predictor["kind"] == "seasonal-naive":
        return make_seasonal_naive_predictions(target_values, origins)
    seasonal_predictions = make_seasonal_member_predictions(target_values, origins, predictor)
    members = np.stack(
        [
            make_last_value_predictions(target_values, origins),
            make_previous_window_predictions(target_values, origins),
            seasonal_predictions,
        ],
        axis=-1,
    )
    return np.sum(members * predictor["weights"][None, :, :], axis=-1).astype(np.float32)


def serialize_anchor_predictor(predictor: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in predictor.items()
    }


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return {"mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error)))}


if TORCH_AVAILABLE:

    class SupervisedWindowDataset(Dataset):
        """Training-only dataset. This is the only Dataset class that receives y."""

        def __init__(self, features: np.ndarray, targets: np.ndarray, origins: np.ndarray, input_size: int):
            self.features = features
            self.targets = targets
            self.origins = origins
            self.input_size = input_size

        def __len__(self) -> int:
            return len(self.origins)

        def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
            origin = int(self.origins[item])
            x = np.ascontiguousarray(self.features[origin - self.input_size : origin])
            y = np.ascontiguousarray(self.targets[origin : origin + HORIZON])
            return torch.from_numpy(x), torch.from_numpy(y)


    class InferenceWindowDataset(Dataset):
        """Inference-only dataset. No future target array is accepted or referenced."""

        def __init__(self, features: np.ndarray, origins: np.ndarray, input_size: int):
            self.features = features
            self.origins = origins
            self.input_size = input_size

        def __len__(self) -> int:
            return len(self.origins)

        def __getitem__(self, item: int) -> torch.Tensor:
            origin = int(self.origins[item])
            x = np.ascontiguousarray(self.features[origin - self.input_size : origin])
            return torch.from_numpy(x)


    class ResidualWindowDataset(Dataset):
        """Training-only PatchTST dataset for leakage-safe anchor residual targets."""

        def __init__(
            self,
            features: np.ndarray,
            target_values: np.ndarray,
            origins: np.ndarray,
            baseline_predictions: np.ndarray,
            input_size: int,
            residual_scale: float,
        ):
            self.features = features
            self.target_values = target_values
            self.origins = origins
            self.baseline_predictions = baseline_predictions
            self.input_size = input_size
            self.residual_scale = residual_scale
            assert self.baseline_predictions.shape == (len(self.origins), HORIZON)

        def __len__(self) -> int:
            return len(self.origins)

        def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
            origin = int(self.origins[item])
            x = np.ascontiguousarray(self.features[origin - self.input_size : origin])
            baseline = self.baseline_predictions[item]
            target = self.target_values[origin : origin + HORIZON]
            residual = np.ascontiguousarray((target - baseline) / self.residual_scale, dtype=np.float32)
            return torch.from_numpy(x), torch.from_numpy(residual)


    class PatchTST(nn.Module):
        """Compact direct PatchTST-style encoder for multivariate input."""

        def __init__(
            self,
            input_size: int,
            num_features: int,
            patch_len: int = 16,
            stride: int = 8,
            d_model: int = 64,
            n_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
            zero_init_head: bool = False,
            window_revin_numeric_features: int = 0,
        ):
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride
            self.window_revin_numeric_features = window_revin_numeric_features
            self.num_patches = (input_size - patch_len) // stride + 1
            projection_features = num_features + 2 * window_revin_numeric_features
            self.patch_projection = nn.Linear(patch_len * projection_features, d_model)
            self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(self.num_patches * d_model, HORIZON))
            nn.init.normal_(self.position_embedding, std=0.02)
            if zero_init_head:
                nn.init.zeros_(self.head[-1].weight)
                nn.init.zeros_(self.head[-1].bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.window_revin_numeric_features:
                count = self.window_revin_numeric_features
                numeric = x[:, :, :count]
                mean = numeric.mean(dim=1, keepdim=True).detach()
                std = torch.sqrt(numeric.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
                normalized = (numeric - mean) / std
                x = torch.cat(
                    [
                        normalized,
                        x[:, :, count:],
                        mean.expand(-1, x.shape[1], -1),
                        std.expand(-1, x.shape[1], -1),
                    ],
                    dim=-1,
                )
            patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            patches = patches.contiguous().reshape(x.shape[0], self.num_patches, -1)
            encoded = self.patch_projection(patches) + self.position_embedding
            return self.head(self.encoder(encoded))


    class DLinear(nn.Module):
        """DLinear with a feature-mixing projection for multivariate-to-OT output."""

        def __init__(self, input_size: int, num_features: int, moving_average: int = 25):
            super().__init__()
            self.moving_average = moving_average
            self.seasonal_linear = nn.Linear(input_size, HORIZON)
            self.trend_linear = nn.Linear(input_size, HORIZON)
            self.feature_projection = nn.Linear(num_features, 1)

        def moving_mean(self, x: torch.Tensor) -> torch.Tensor:
            pad = (self.moving_average - 1) // 2
            padded = torch.cat([x[:, :1].repeat(1, pad, 1), x, x[:, -1:].repeat(1, pad, 1)], dim=1)
            return F.avg_pool1d(padded.transpose(1, 2), kernel_size=self.moving_average, stride=1).transpose(1, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            trend = self.moving_mean(x)
            seasonal = x - trend
            forecast = self.seasonal_linear(seasonal.transpose(1, 2)) + self.trend_linear(trend.transpose(1, 2))
            return self.feature_projection(forecast.transpose(1, 2)).squeeze(-1)


    class NBeatsBlock(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int, dropout: float):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.backcast = nn.Linear(hidden_size, input_dim)
            self.forecast = nn.Linear(hidden_size, HORIZON)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.layers(x)
            return self.backcast(hidden), self.forecast(hidden)


    class NBeats(nn.Module):
        def __init__(self, input_size: int, num_features: int, hidden_size: int = 256, dropout: float = 0.1):
            super().__init__()
            input_dim = input_size * num_features
            self.blocks = nn.ModuleList([NBeatsBlock(input_dim, hidden_size, dropout) for _ in range(3)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x.flatten(start_dim=1)
            forecast = residual.new_zeros((residual.shape[0], HORIZON))
            for block in self.blocks:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast
            return forecast


    class NHiTSBlock(nn.Module):
        def __init__(
            self,
            input_size: int,
            num_features: int,
            pool_size: int,
            horizon_downsample: int,
            hidden_size: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_size = input_size
            self.num_features = num_features
            self.pool_size = pool_size
            self.pooled_size = math.ceil(input_size / pool_size)
            self.knot_count = math.ceil(HORIZON / horizon_downsample)
            input_dim = self.pooled_size * num_features
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.backcast = nn.Linear(hidden_size, input_dim)
            self.forecast = nn.Linear(hidden_size, self.knot_count)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            pooled = F.avg_pool1d(
                x.transpose(1, 2),
                kernel_size=self.pool_size,
                stride=self.pool_size,
                ceil_mode=True,
            )
            hidden = self.layers(pooled.flatten(start_dim=1))
            backcast = self.backcast(hidden).reshape(x.shape[0], self.num_features, self.pooled_size)
            backcast = F.interpolate(backcast, size=self.input_size, mode="linear", align_corners=False).transpose(1, 2)
            forecast = self.forecast(hidden).unsqueeze(1)
            forecast = F.interpolate(forecast, size=HORIZON, mode="linear", align_corners=False).squeeze(1)
            return backcast, forecast


    class NHiTS(nn.Module):
        def __init__(self, input_size: int, num_features: int, dropout: float = 0.1):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    NHiTSBlock(input_size, num_features, pool_size=24, horizon_downsample=24, dropout=dropout),
                    NHiTSBlock(input_size, num_features, pool_size=4, horizon_downsample=4, dropout=dropout),
                    NHiTSBlock(input_size, num_features, pool_size=1, horizon_downsample=1, dropout=dropout),
                ]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            forecast = x.new_zeros((x.shape[0], HORIZON))
            for block in self.blocks:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast
            return forecast


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for neural model training. Kaggle normally includes it. "
            "Use --validate-only for the NumPy leakage check in a minimal local environment."
        )


def device_name() -> str:
    require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def predict_torch_model(
    model: nn.Module,
    data: PreparedData,
    origins: np.ndarray,
    input_size: int,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Prediction path accepts X and origins only. It never constructs test y."""
    inference_dataset = InferenceWindowDataset(data.features_scaled, origins, input_size)
    loader = make_loader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in loader:
            y_scaled = model(x.to(device))
            predictions.append(y_scaled.detach().cpu().numpy())
    scaled = np.concatenate(predictions, axis=0)
    return (scaled * data.target_std + data.target_mean).astype(np.float32)


def train_torch_model(
    name: str,
    model: nn.Module,
    data: PreparedData,
    input_size: int,
    seed: int,
    learning_rate: float,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    model = model.to(device)
    train_dataset = SupervisedWindowDataset(
        data.features_scaled,
        data.target_scaled,
        data.split.train_origins,
        input_size,
    )
    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    val_true = make_supervised_targets(data.target_raw, data.split.val_origins)
    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    early_stopped = False
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(x)
            train_count += len(x)

        val_pred = predict_torch_model(model, data, data.split.val_origins, input_size, device, batch_size, num_workers)
        val_metric = metric(val_true, val_pred)
        train_scaled_mse = train_loss_sum / max(1, train_count)
        print(
            f"{name} seed={seed} epoch={epoch:02d}/{epochs}: "
            f"train_scaled_mse={train_scaled_mse:.6f}, "
            f"val_mse={val_metric['mse']:.6f}, val_mae={val_metric['mae']:.6f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_scaled_mse": float(train_scaled_mse),
                "val_mse": float(val_metric["mse"]),
                "val_mae": float(val_metric["mae"]),
            }
        )
        if val_metric["mse"] < best_mse - 1e-8:
            best_mse = val_metric["mse"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"{name} seed={seed}: early stopping at epoch {epoch}.")
                early_stopped = True
                break

    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model = model.to(device)
    val_pred = predict_torch_model(model, data, data.split.val_origins, input_size, device, batch_size, num_workers)
    test_pred = predict_torch_model(model, data, data.split.test_origins, input_size, device, batch_size, num_workers)
    return val_pred, test_pred, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mse": float(best_mse),
        "early_stopped": early_stopped,
        "epochs": history,
    }


def train_seed_family(
    name: str,
    builder: Callable[[], nn.Module],
    data: PreparedData,
    input_size: int,
    seeds: tuple[int, ...],
    learning_rate: float,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    validation_predictions = []
    test_predictions = []
    histories = []
    for seed in seeds:
        set_seed(seed)
        model = builder()
        val_pred, test_pred, history = train_torch_model(
            name=name,
            model=model,
            data=data,
            input_size=input_size,
            seed=seed,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        validation_predictions.append(val_pred)
        test_predictions.append(test_pred)
        histories.append(history)
    return np.mean(validation_predictions, axis=0), np.mean(test_predictions, axis=0), histories


def make_residual_targets(
    target_values: np.ndarray,
    origins: np.ndarray,
    baseline_predictions: np.ndarray,
) -> np.ndarray:
    true = make_supervised_targets(target_values, origins)
    assert baseline_predictions.shape == true.shape
    return (true - baseline_predictions).astype(np.float32)


def fit_residual_scale(
    target_values: np.ndarray,
    origins: np.ndarray,
    baseline_predictions: np.ndarray,
) -> float:
    residual = make_residual_targets(target_values, origins, baseline_predictions)
    scale = float(np.sqrt(np.mean(np.asarray(residual, dtype=np.float64) ** 2)))
    return max(scale, 1e-6)


def train_residual_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    ema: "ModelEMA | None" = None,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    loss_sum = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        loss_sum += float(loss.detach().cpu()) * len(x)
        count += len(x)
    return loss_sum / max(1, count)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("--patch-ema-decay must be between 0 and 1.")
        self.decay = float(decay)
        self.shadow = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            value_cpu = value.detach().cpu()
            if torch.is_floating_point(value_cpu):
                self.shadow[key].mul_(self.decay).add_(value_cpu, alpha=1.0 - self.decay)
            else:
                self.shadow[key] = value_cpu.clone()

    def state_dict(self) -> dict:
        return {key: value.clone() for key, value in self.shadow.items()}


def predict_patchtst_residual(
    model: nn.Module,
    data: PreparedData,
    origins: np.ndarray,
    input_size: int,
    residual_scale: float,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Predict residuals from X only. Test-period future OT is never accepted."""
    inference_dataset = InferenceWindowDataset(data.features_scaled, origins, input_size)
    loader = make_loader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in loader:
            scaled_residual = model(x.to(device))
            predictions.append(scaled_residual.detach().cpu().numpy())
    return (np.concatenate(predictions, axis=0) * residual_scale).astype(np.float32)


def predict_patchtst_residual_with_state(
    model: nn.Module,
    state: dict,
    data: PreparedData,
    origins: np.ndarray,
    input_size: int,
    residual_scale: float,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    current_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    model = model.to(device)
    prediction = predict_patchtst_residual(
        model,
        data,
        origins,
        input_size,
        residual_scale,
        device,
        batch_size,
        num_workers,
    )
    model.load_state_dict(current_state)
    model = model.to(device)
    return prediction


def patch_window_revin_numeric_features(data: PreparedData, enabled: bool) -> int:
    if not enabled:
        return 0
    return 1 if data.feature_names == [TARGET_COL] else len(FEATURE_COLS)


def train_patchtst_residual_seed(
    name: str,
    data: PreparedData,
    input_size: int,
    seed: int,
    train_all_hour_origins: np.ndarray,
    train_midnight_origins: np.ndarray,
    train_all_hour_baseline: np.ndarray,
    train_midnight_baseline: np.ndarray,
    val_baseline: np.ndarray,
    residual_scale: float,
    pretrain_epochs: int,
    finetune_epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    device: str,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.2,
    window_revin: bool = False,
    ema_enabled: bool = False,
    ema_decay: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    set_seed(seed)
    model = PatchTST(
        input_size=input_size,
        num_features=len(data.feature_names),
        dropout=dropout,
        zero_init_head=True,
        window_revin_numeric_features=patch_window_revin_numeric_features(data, window_revin),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ema = ModelEMA(model, ema_decay) if ema_enabled else None
    pretrain_loader = make_loader(
        ResidualWindowDataset(
            data.features_scaled,
            data.target_raw,
            train_all_hour_origins,
            train_all_hour_baseline,
            input_size,
            residual_scale,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    finetune_loader = make_loader(
        ResidualWindowDataset(
            data.features_scaled,
            data.target_raw,
            train_midnight_origins,
            train_midnight_baseline,
            input_size,
            residual_scale,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    history = []
    for epoch in range(1, pretrain_epochs + 1):
        train_mse = train_residual_epoch(model, pretrain_loader, optimizer, device, ema)
        print(f"{name} seed={seed} pretrain={epoch:02d}/{pretrain_epochs}: scaled_residual_mse={train_mse:.6f}")
        history.append({"phase": "pretrain", "epoch": epoch, "train_scaled_residual_mse": float(train_mse)})

    val_true = make_supervised_targets(data.target_raw, data.split.val_origins)
    best_mse = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_ema_mse = float("inf")
    best_ema_epoch = 0
    best_ema_state = ema.state_dict() if ema is not None else None
    stale_epochs = 0
    early_stopped = False
    for epoch in range(1, finetune_epochs + 1):
        train_mse = train_residual_epoch(model, finetune_loader, optimizer, device, ema)
        val_residual = predict_patchtst_residual(
            model,
            data,
            data.split.val_origins,
            input_size,
            residual_scale,
            device,
            batch_size,
            num_workers,
        )
        val_metric = metric(val_true, val_baseline + val_residual)
        ema_metric = None
        if ema is not None:
            ema_val_residual = predict_patchtst_residual_with_state(
                model,
                ema.state_dict(),
                data,
                data.split.val_origins,
                input_size,
                residual_scale,
                device,
                batch_size,
                num_workers,
            )
            ema_metric = metric(val_true, val_baseline + ema_val_residual)
            if ema_metric["mse"] < best_ema_mse - 1e-8:
                best_ema_mse = ema_metric["mse"]
                best_ema_epoch = epoch
                best_ema_state = ema.state_dict()
        print(
            f"{name} seed={seed} finetune={epoch:02d}/{finetune_epochs}: "
            f"scaled_residual_mse={train_mse:.6f}, val_mse_lambda1={val_metric['mse']:.6f}, "
            f"val_mae_lambda1={val_metric['mae']:.6f}"
            + (
                f", ema_val_mse_lambda1={ema_metric['mse']:.6f}, ema_val_mae_lambda1={ema_metric['mae']:.6f}"
                if ema_metric is not None
                else ""
            )
        )
        row = {
            "phase": "finetune",
            "epoch": epoch,
            "train_scaled_residual_mse": float(train_mse),
            "val_mse_lambda1": float(val_metric["mse"]),
            "val_mae_lambda1": float(val_metric["mae"]),
        }
        if ema_metric is not None:
            row["ema_val_mse_lambda1"] = float(ema_metric["mse"])
            row["ema_val_mae_lambda1"] = float(ema_metric["mae"])
        history.append(row)
        if val_metric["mse"] < best_mse - 1e-8:
            best_mse = val_metric["mse"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"{name} seed={seed}: early stopping at finetune epoch {epoch}.")
                early_stopped = True
                break

    model.load_state_dict(best_state)
    model = model.to(device)
    val_residual = predict_patchtst_residual(
        model,
        data,
        data.split.val_origins,
        input_size,
        residual_scale,
        device,
        batch_size,
        num_workers,
    )
    test_residual = predict_patchtst_residual(
        model,
        data,
        data.split.test_origins,
        input_size,
        residual_scale,
        device,
        batch_size,
        num_workers,
    )
    ema_val_residual = None
    ema_test_residual = None
    if ema_enabled and best_ema_state is not None:
        ema_val_residual = predict_patchtst_residual_with_state(
            model,
            best_ema_state,
            data,
            data.split.val_origins,
            input_size,
            residual_scale,
            device,
            batch_size,
            num_workers,
        )
        ema_test_residual = predict_patchtst_residual_with_state(
            model,
            best_ema_state,
            data,
            data.split.test_origins,
            input_size,
            residual_scale,
            device,
            batch_size,
            num_workers,
        )
    return val_residual, test_residual, ema_val_residual, ema_test_residual, {
        "seed": seed,
        "input_size": input_size,
        "residual_scale": residual_scale,
        "window_revin": bool(window_revin),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "ema_enabled": bool(ema_enabled),
        "ema_decay": float(ema_decay),
        "pretrain_epochs": pretrain_epochs,
        "best_finetune_epoch": best_epoch,
        "best_val_mse_lambda1": float(best_mse),
        "best_ema_finetune_epoch": int(best_ema_epoch),
        "best_ema_val_mse_lambda1": float(best_ema_mse) if ema_enabled else None,
        "early_stopped": early_stopped,
        "epochs": history,
    }


def fit_patchtst_residual_refit_seed(
    name: str,
    data: PreparedData,
    input_size: int,
    seed: int,
    train_all_hour_origins: np.ndarray,
    train_midnight_origins: np.ndarray,
    train_all_hour_baseline: np.ndarray,
    train_midnight_baseline: np.ndarray,
    residual_scale: float,
    pretrain_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.2,
    window_revin: bool = False,
    ema_enabled: bool = False,
    ema_decay: float = 0.995,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    set_seed(seed)
    model = PatchTST(
        input_size=input_size,
        num_features=len(data.feature_names),
        dropout=dropout,
        zero_init_head=True,
        window_revin_numeric_features=patch_window_revin_numeric_features(data, window_revin),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ema = ModelEMA(model, ema_decay) if ema_enabled else None
    pretrain_loader = make_loader(
        ResidualWindowDataset(
            data.features_scaled,
            data.target_raw,
            train_all_hour_origins,
            train_all_hour_baseline,
            input_size,
            residual_scale,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    finetune_loader = make_loader(
        ResidualWindowDataset(
            data.features_scaled,
            data.target_raw,
            train_midnight_origins,
            train_midnight_baseline,
            input_size,
            residual_scale,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    history = []
    for epoch in range(1, pretrain_epochs + 1):
        train_mse = train_residual_epoch(model, pretrain_loader, optimizer, device, ema)
        print(f"{name} seed={seed} refit-pretrain={epoch:02d}/{pretrain_epochs}: scaled_residual_mse={train_mse:.6f}")
        history.append({"phase": "refit-pretrain", "epoch": epoch, "train_scaled_residual_mse": float(train_mse)})
    for epoch in range(1, finetune_epochs + 1):
        train_mse = train_residual_epoch(model, finetune_loader, optimizer, device, ema)
        print(f"{name} seed={seed} refit-finetune={epoch:02d}/{finetune_epochs}: scaled_residual_mse={train_mse:.6f}")
        history.append({"phase": "refit-finetune", "epoch": epoch, "train_scaled_residual_mse": float(train_mse)})

    test_residual = predict_patchtst_residual(
        model,
        data,
        data.split.test_origins,
        input_size,
        residual_scale,
        device,
        batch_size,
        num_workers,
    )
    ema_test_residual = None
    if ema is not None:
        ema_test_residual = predict_patchtst_residual_with_state(
            model,
            ema.state_dict(),
            data,
            data.split.test_origins,
            input_size,
            residual_scale,
            device,
            batch_size,
            num_workers,
        )
    return test_residual, ema_test_residual, {
        "seed": seed,
        "input_size": input_size,
        "residual_scale": residual_scale,
        "window_revin": bool(window_revin),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "ema_enabled": bool(ema_enabled),
        "ema_decay": float(ema_decay),
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
        "epochs": history,
    }


def gbm_origin_features(raw: RawData, origin: int) -> list[float]:
    """Compute past-observation features once per origin."""
    numeric = raw.numeric_values
    features: list[float] = []
    for lag in GBM_LAGS:
        features.extend(float(value) for value in numeric[origin - lag])
    for window in GBM_ROLL_WINDOWS:
        history = numeric[origin - window : origin]
        features.extend(float(value) for value in history.mean(axis=0))
        features.extend(float(value) for value in history.std(axis=0))
        features.extend(float(value) for value in history.min(axis=0))
        features.extend(float(value) for value in history.max(axis=0))
    features.extend(datetime_features(raw.dates[origin]))
    return features


def gbm_origin_horizon_features(raw: RawData, origin: int, horizon: int, origin_features: list[float] | None = None) -> np.ndarray:
    """Use numeric observations before origin plus known calendar features only."""
    features = list(origin_features) if origin_features is not None else gbm_origin_features(raw, origin)
    features.extend(datetime_features(raw.dates[origin] + timedelta(hours=horizon)))
    features.extend(
        [
            float(horizon),
            math.sin(2.0 * math.pi * horizon / HORIZON),
            math.cos(2.0 * math.pi * horizon / HORIZON),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def build_gbm_feature_matrix(raw: RawData, origins: np.ndarray) -> np.ndarray:
    """Inference-safe GBM feature builder. Future OT is not accepted or read."""
    rows = []
    for origin in origins:
        origin = int(origin)
        shared_features = gbm_origin_features(raw, origin)
        rows.extend(gbm_origin_horizon_features(raw, origin, horizon, shared_features) for horizon in range(HORIZON))
    return np.stack(rows).astype(np.float32)


def build_tree_anchor_feature_matrix(
    raw: RawData,
    target_values: np.ndarray,
    origins: np.ndarray,
    heuristic_predictor: dict,
) -> np.ndarray:
    """Build inference-safe tree features from past observations and anchor members only."""
    heuristic = make_anchor_predictions(target_values, origins, heuristic_predictor)
    last_value = make_last_value_predictions(target_values, origins)
    previous_window = make_previous_window_predictions(target_values, origins)
    previous_week = make_previous_week_predictions(target_values, origins)
    previous_two_week = make_previous_two_week_predictions(target_values, origins)
    anchor_features = np.stack(
        [
            heuristic,
            last_value,
            previous_window,
            previous_week,
            previous_two_week,
            heuristic - last_value,
            heuristic - previous_window,
            heuristic - previous_week,
            previous_week - previous_two_week,
        ],
        axis=-1,
    ).reshape(-1, 9)
    return np.concatenate([build_gbm_feature_matrix(raw, origins), anchor_features], axis=1).astype(np.float32)


def make_tree_anchor_train_origins(last_origin: int, step_hours: int) -> np.ndarray:
    if step_hours < 1:
        raise ValueError("--tree-anchor-train-step-hours must be at least 1.")
    first_origin = max(max(GBM_LAGS), max(GBM_ROLL_WINDOWS), 24 * 14)
    origins = np.arange(first_origin, last_origin + 1, step_hours, dtype=np.int64)
    if len(origins) == 0:
        raise ValueError("tree anchor has no training origins.")
    return origins


def fit_tree_anchor_model(
    data: PreparedData,
    heuristic_predictor: dict,
    last_origin: int,
    residual_lambda: float,
    train_step_hours: int,
    estimator_count: int,
) -> dict:
    """Fit LightGBM residuals with an internal chronological validation split."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required for --tree-anchor.") from exc

    all_origins = make_tree_anchor_train_origins(last_origin, train_step_hours)
    internal_val_start = last_origin - 21 * 24
    train_origins = all_origins[all_origins + HORIZON <= internal_val_start]
    val_origins = all_origins[all_origins >= internal_val_start]
    if len(train_origins) == 0 or len(val_origins) == 0:
        raise ValueError("tree anchor internal split is empty.")
    assert int(train_origins[-1]) + HORIZON <= int(val_origins[0])

    train_baseline = make_anchor_predictions(data.target_raw, train_origins, heuristic_predictor)
    val_baseline = make_anchor_predictions(data.target_raw, val_origins, heuristic_predictor)
    train_x = build_tree_anchor_feature_matrix(data.raw, data.target_raw, train_origins, heuristic_predictor)
    val_x = build_tree_anchor_feature_matrix(data.raw, data.target_raw, val_origins, heuristic_predictor)
    train_residual = make_residual_targets(data.target_raw, train_origins, train_baseline)
    val_residual = make_residual_targets(data.target_raw, val_origins, val_baseline)
    residual_scale = train_residual.std(axis=0).astype(np.float32)
    residual_scale = np.where(residual_scale < 1e-3, 1.0, residual_scale).astype(np.float32)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=estimator_count,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
    )
    callbacks = [lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=50, verbose=False)]
    model.fit(
        train_x,
        train_residual.reshape(-1),
        eval_set=[(val_x, val_residual.reshape(-1))],
        eval_metric="l2",
        callbacks=callbacks,
    )
    return {
        "name": f"lightgbm_residual_anchor_lambda{residual_lambda:.2f}",
        "kind": "tree-anchor",
        "heuristic_predictor": heuristic_predictor,
        "model": model,
        "lambda": float(residual_lambda),
        "residual_scale": residual_scale,
        "train_origins": train_origins,
        "internal_val_origins": val_origins,
        "train_step_hours": int(train_step_hours),
        "estimators": int(estimator_count),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or estimator_count),
    }


def predict_tree_anchor_residual(
    data: PreparedData,
    origins: np.ndarray,
    tree_anchor: dict,
) -> np.ndarray:
    """Predict LightGBM residuals in bounded chunks without constructing future labels."""
    predictions = []
    for start in range(0, len(origins), TREE_ANCHOR_PREDICT_CHUNK_ORIGINS):
        chunk = origins[start : start + TREE_ANCHOR_PREDICT_CHUNK_ORIGINS]
        features = build_tree_anchor_feature_matrix(
            data.raw,
            data.target_raw,
            chunk,
            tree_anchor["heuristic_predictor"],
        )
        predictions.append(
            tree_anchor["model"].booster_.predict(
                features,
                num_iteration=tree_anchor["best_iteration"],
            ).reshape(len(chunk), HORIZON).astype(np.float32)
        )
    residual = np.concatenate(predictions, axis=0)
    clip = 2.0 * np.asarray(tree_anchor["residual_scale"], dtype=np.float32)[None, :]
    return np.clip(residual, -clip, clip).astype(np.float32)


def make_tree_anchor_predictions(data: PreparedData, origins: np.ndarray, tree_anchor: dict) -> np.ndarray:
    heuristic = make_anchor_predictions(data.target_raw, origins, tree_anchor["heuristic_predictor"])
    residual = predict_tree_anchor_residual(data, origins, tree_anchor)
    return (heuristic + float(tree_anchor["lambda"]) * residual).astype(np.float32)


def serialize_tree_anchor(tree_anchor: dict | None) -> dict | None:
    if tree_anchor is None:
        return None
    return {
        "name": tree_anchor["name"],
        "kind": tree_anchor["kind"],
        "lambda": float(tree_anchor["lambda"]),
        "train_step_hours": int(tree_anchor["train_step_hours"]),
        "estimators": int(tree_anchor["estimators"]),
        "best_iteration": int(tree_anchor["best_iteration"]),
        "train_samples": int(len(tree_anchor["train_origins"])),
        "internal_validation_samples": int(len(tree_anchor["internal_val_origins"])),
        "heuristic_predictor": serialize_anchor_predictor(tree_anchor["heuristic_predictor"]),
    }


def fit_pipeline_anchor(
    data: PreparedData,
    heuristic_fit_origins: np.ndarray,
    last_origin: int,
    args: argparse.Namespace,
    quick: bool = False,
) -> tuple[dict, dict | None]:
    heuristic_predictor = fit_anchor_predictor(data.target_raw, heuristic_fit_origins, args.anchor)
    if not args.tree_anchor:
        return heuristic_predictor, None
    estimator_count = min(args.tree_anchor_estimators, 80) if quick else args.tree_anchor_estimators
    tree_anchor = fit_tree_anchor_model(
        data,
        heuristic_predictor,
        last_origin=last_origin,
        residual_lambda=args.tree_anchor_lambda,
        train_step_hours=args.tree_anchor_train_step_hours,
        estimator_count=estimator_count,
    )
    return heuristic_predictor, tree_anchor


def make_pipeline_anchor_predictions(
    data: PreparedData,
    origins: np.ndarray,
    heuristic_predictor: dict,
    tree_anchor: dict | None,
) -> np.ndarray:
    if tree_anchor is None:
        return make_anchor_predictions(data.target_raw, origins, heuristic_predictor)
    return make_tree_anchor_predictions(data, origins, tree_anchor)


def pipeline_anchor_name(heuristic_predictor: dict, tree_anchor: dict | None) -> str:
    if tree_anchor is None:
        return heuristic_predictor["name"]
    return f"{heuristic_predictor['name']} + {tree_anchor['name']}"


def train_optional_gbm(
    data: PreparedData,
    quick: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import lightgbm as lgb
    except ImportError:
        print("GBM optional: skipped because lightgbm is not installed.")
        return None

    stage("5. Train optional LightGBM lag-feature model")
    train_x = build_gbm_feature_matrix(data.raw, data.split.train_origins)
    val_x = build_gbm_feature_matrix(data.raw, data.split.val_origins)
    test_x = build_gbm_feature_matrix(data.raw, data.split.test_origins)
    train_y = make_supervised_targets(data.target_raw, data.split.train_origins).reshape(-1)
    val_y = make_supervised_targets(data.target_raw, data.split.val_origins).reshape(-1)
    estimator_count = 80 if quick else 800
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=estimator_count,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    callbacks = [lgb.log_evaluation(period=0)]
    if not quick:
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric="l2", callbacks=callbacks)
    val_pred = model.predict(val_x).reshape(len(data.split.val_origins), HORIZON).astype(np.float32)
    test_pred = model.predict(test_x).reshape(len(data.split.test_origins), HORIZON).astype(np.float32)
    return val_pred, test_pred


def weighted_prediction(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    missing = [name for name in weights if name not in predictions]
    if missing:
        raise ValueError(f"Missing predictions for ensemble members: {missing}")
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ensemble weights must sum to 1.0, found {total}.")
    return sum(float(weight) * predictions[name] for name, weight in weights.items())


def cap_and_renormalize(weights: np.ndarray, max_weight: float = 0.5) -> np.ndarray:
    """Normalize inverse-MSE weights while preserving the requested final cap."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0 or np.any(weights < 0):
        raise ValueError("weights must be a non-empty one-dimensional non-negative array.")
    if len(weights) * max_weight < 1.0:
        raise ValueError("max_weight is too small to produce normalized weights.")
    if weights.sum() <= 0:
        raise ValueError("At least one weight must be positive.")

    result = np.zeros_like(weights)
    active = np.ones(len(weights), dtype=bool)
    remaining_mass = 1.0
    while np.any(active):
        active_weights = weights[active]
        if active_weights.sum() <= 0:
            result[active] = remaining_mass / int(active.sum())
            break
        normalized = active_weights / active_weights.sum() * remaining_mass
        over_cap = normalized > max_weight
        if not np.any(over_cap):
            result[active] = normalized
            break
        active_indices = np.where(active)[0]
        capped_indices = active_indices[over_cap]
        result[capped_indices] = max_weight
        active[capped_indices] = False
        remaining_mass -= max_weight * len(capped_indices)
    result = result / result.sum()
    assert float(result.max()) <= max_weight + 1e-10
    return result


def inverse_mse_weights(
    validation_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict[str, float]:
    names = list(validation_predictions)
    mse_values = np.asarray([metric(y_true, validation_predictions[name])["mse"] for name in names], dtype=np.float64)
    weights = 1.0 / (mse_values + 1e-8)
    weights = cap_and_renormalize(weights, max_weight=0.5)
    return {name: float(weight) for name, weight in zip(names, weights)}


def assert_prediction_shapes(predictions: dict[str, np.ndarray], expected_rows: int) -> None:
    expected_shape = (expected_rows, HORIZON)
    for name, prediction in predictions.items():
        if prediction.shape != expected_shape:
            raise ValueError(f"{name} prediction shape must be {expected_shape}, found {prediction.shape}.")
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} prediction contains NaN or infinite values.")
        print(f"prediction shape assertion: {name} -> {prediction.shape}")


def save_submission(raw: RawData, predictions: np.ndarray, output_path: Path) -> None:
    expected_shape = (len(raw.sample_ids), HORIZON)
    if predictions.shape != expected_shape:
        raise ValueError(f"Submission prediction shape must be {expected_shape}, found {predictions.shape}.")
    expected_columns = [raw.sample_id_col, *[f"T{i}" for i in range(HORIZON)]]
    assert expected_columns == raw.sample_columns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(expected_columns)
        for sample_id, row in zip(raw.sample_ids, predictions):
            writer.writerow([sample_id, *[float(value) for value in row]])

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        written_rows = list(reader)
    assert written_rows[0] == expected_columns
    assert len(written_rows) - 1 == len(raw.sample_ids)
    print(f"submit.csv schema assertion: {written_rows[0][:4]} ... {written_rows[0][-3:]}")
    print(f"submit.csv row assertion: {len(written_rows) - 1} == sample_submit.csv rows {len(raw.sample_ids)}")
    print(f"submit.csv saved: {output_path.resolve()}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_run_report(
    args: argparse.Namespace,
    data: PreparedData,
    mode: str,
    results: dict[str, dict[str, float]],
    best_single: str,
    best_overall: str,
    selected_weights: dict[str, float],
    used_models: list[str],
    training_history: dict[str, list[dict]] | None = None,
) -> None:
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "command": sys.argv,
        "settings": {
            "input_mode": "multivariate + time features" if not args.univariate_input else "univariate OT only",
            "patch_seeds": list(parse_seed_list(args.patch_seeds)),
            "epochs": 1 if args.quick else args.epochs,
            "patience": 1 if args.quick else args.patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "validation_fraction": args.validation_fraction,
            "skip_gbm": bool(args.skip_gbm),
            "quick": bool(args.quick),
            "validate_only": bool(args.validate_only),
        },
        "split": {
            "train_samples": int(len(data.split.train_origins)),
            "validation_samples": int(len(data.split.val_origins)),
            "test_samples": int(len(data.split.test_origins)),
            "last_supervised_origin": data.raw.dates[data.split.test_start_idx - HORIZON].isoformat(sep=" "),
            "last_supervised_y_end": LAST_ALLOWED_TARGET_END.isoformat(sep=" "),
            "scaler_fit_end_exclusive": data.raw.dates[data.split.scaler_fit_end_idx].isoformat(sep=" "),
            "test_input_rule": "each X window ends at its submission timestamp minus one hour",
        },
        "models_used": used_models,
        "training_history": training_history or {},
        "validation_results": results,
        "best_single_model": best_single,
        "best_overall_predictor": best_overall,
        "selected_weights": selected_weights,
        "submission": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "rows": len(data.raw.sample_ids),
            "columns": len(data.raw.sample_columns),
        },
        "kaggle_public_mse": None,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"run report saved: {args.report_output.resolve()}")
    print("share this JSON report and the Kaggle public MSE after submission.")


def print_validation_results(results: dict[str, dict[str, float]], best_single: str, best_overall: str) -> None:
    print("\nValidation Results")
    print("------------------")
    order = [
        "PatchTST",
        "DLinear/NLinear",
        "N-BEATS",
        "N-HiTS",
        "Seasonal Naive",
        "GBM optional",
        "Fixed Ensemble",
        "Fixed Ensemble with GBM",
        "Inverse-MSE Weighted Ensemble",
        "Inverse-MSE Weighted Ensemble with GBM",
    ]
    for name in order:
        if name in results:
            print(f"{name}: MSE = {results[name]['mse']:.6f}, MAE = {results[name]['mae']:.6f}")
    print(f"Best Single Model: {best_single}")
    print(f"Best Overall: {best_overall}")


def print_summary(
    used_models: list[str],
    best_single: str,
    best_overall: str,
    selected_weights: dict[str, float],
) -> None:
    print("\nReport Summary")
    print("--------------")
    print(f"Models used: {', '.join(used_models)}")
    print(
        "PatchTST is the main model because patch-based attention can represent medium- and long-range "
        "patterns efficiently; the 512-hour and 336-hour variants are blended at 0.6 and 0.4."
    )
    print(
        "DLinear is included because its trend/seasonal decomposition adds a low-complexity linear "
        "inductive bias that is complementary to attention models."
    )
    print(
        "Seasonal Naive is included because repeating the last observed 24 hours is a leakage-safe, "
        "stable hourly baseline that can reduce ensemble variance."
    )
    print(
        "Leakage rule: train/validation y_end never exceeds 2018-01-31 23:00:00; scaler fitting stops "
        "before validation; each test X ends at its own timestamp minus one hour."
    )
    print(f"Best validation single model: {best_single}")
    print(f"Best validation overall predictor: {best_overall}")
    print(f"Final submission predictor: {best_overall}")
    print(f"Final submission weights: {selected_weights}")


def parse_int_list(value: str, option_name: str) -> tuple[int, ...]:
    numbers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not numbers:
        raise ValueError(f"{option_name} must contain at least one integer.")
    return numbers


def parse_float_list(value: str, option_name: str) -> tuple[float, ...]:
    numbers = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not numbers:
        raise ValueError(f"{option_name} must contain at least one number.")
    return numbers


def parse_block_lambda_candidates(value: str, option_name: str) -> tuple[tuple[float, ...], ...]:
    candidates = []
    for raw_candidate in value.split(";"):
        raw_candidate = raw_candidate.strip()
        if not raw_candidate:
            continue
        parts = tuple(float(part.strip()) for part in raw_candidate.split(",") if part.strip())
        if len(parts) != 4:
            raise ValueError(f"{option_name} candidates must contain exactly four comma-separated values.")
        if any(part < 0.0 for part in parts):
            raise ValueError(f"{option_name} candidates must be non-negative.")
        candidates.append(parts)
    if not candidates:
        raise ValueError(f"{option_name} must contain at least one candidate.")
    return tuple(dict.fromkeys(candidates))


def horizon_block_slices() -> tuple[tuple[int, int], ...]:
    return tuple((start, start + 24) for start in range(0, HORIZON, 24))


def smooth_block_lambda_candidates(values: tuple[float, ...], max_adjacent_diff: float) -> tuple[tuple[float, ...], ...]:
    values = tuple(sorted(set(float(value) for value in values)))
    if any(value < 0.0 for value in values):
        raise ValueError("--block-lambda-values must be non-negative.")
    candidates = []
    for a in values:
        for b in values:
            for c in values:
                for d in values:
                    candidate = (a, b, c, d)
                    if any(abs(candidate[idx] - candidate[idx - 1]) > max_adjacent_diff + 1e-12 for idx in range(1, 4)):
                        continue
                    if len(set(candidate)) <= 1:
                        continue
                    candidates.append(candidate)
    return tuple(candidates)


def merge_block_lambda_candidates(*groups: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    merged = []
    seen = set()
    for group in groups:
        for candidate in group:
            normalized = tuple(float(value) for value in candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return tuple(merged)


def apply_residual_lambda(
    baseline: np.ndarray,
    residual: np.ndarray,
    residual_lambda: float | tuple[float, ...],
) -> np.ndarray:
    if isinstance(residual_lambda, tuple):
        if len(residual_lambda) != 4:
            raise ValueError("block residual lambda must contain four values.")
        prediction = baseline.copy()
        for block_lambda, (start, end) in zip(residual_lambda, horizon_block_slices()):
            prediction[:, start:end] = baseline[:, start:end] + float(block_lambda) * residual[:, start:end]
        return prediction.astype(np.float32)
    return (baseline + float(residual_lambda) * residual).astype(np.float32)


def lambda_label(row: dict) -> str:
    if row.get("lambda_kind") == "block":
        values = ",".join(f"{value:.2f}" for value in row["block_lambdas"])
        return f"block[{values}]"
    return f"{float(row['lambda']):.2f}"


def lambda_is_active(row: dict) -> bool:
    if row.get("lambda_kind") == "block":
        return any(float(value) > 0.0 for value in row["block_lambdas"])
    return float(row["lambda"]) > 0.0


def lambda_complexity(row: dict) -> tuple[int, float]:
    if row.get("lambda_kind") == "block":
        values = tuple(float(value) for value in row["block_lambdas"])
        return (1, max(values) - min(values))
    return (0, 0.0)


def lambda_spec_from_row(row: dict) -> float | tuple[float, ...]:
    if row.get("lambda_kind") == "block":
        return tuple(float(value) for value in row["block_lambdas"])
    return float(row["lambda"])


def lambda_values_for_report(spec: float | tuple[float, ...]) -> dict:
    if isinstance(spec, tuple):
        return {"lambda_kind": "block", "lambda": None, "block_lambdas": list(spec), "lambda_label": "block[" + ",".join(f"{v:.2f}" for v in spec) + "]"}
    return {"lambda_kind": "scalar", "lambda": float(spec), "block_lambdas": None, "lambda_label": f"{float(spec):.2f}"}


def save_json_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"run report saved: {output_path.resolve()}")


def derived_submission_path(output_path: Path, suffix: str) -> Path:
    if output_path.name == "submit.csv":
        return output_path.with_name(f"submit_{suffix}.csv")
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def select_provisional_anchor(anchor_results: list[dict], tolerance: float = 0.005) -> dict:
    """Prefer a simpler recipe when it is within 0.5% of the holdout minimum."""
    best_mse = min(row["mse"] for row in anchor_results)
    near_best = [row for row in anchor_results if row["mse"] <= best_mse * (1.0 + tolerance)]
    return min(near_best, key=lambda row: (row["complexity"], row["mse"], row["name"]))


def print_anchor_results(anchor_results: list[dict], selected: dict) -> None:
    print("Anchor Candidate Validation Results")
    print("-----------------------------------")
    print(f"{'name':48s} {'complexity':>10s} {'mse':>12s} {'mae':>12s}")
    for row in anchor_results:
        marker = " <- provisional" if row["recipe"] == selected["recipe"] else ""
        print(f"{row['name']:48s} {row['complexity']:10d} {row['mse']:12.6f} {row['mae']:12.6f}{marker}")


def run_anchor_analysis_only(data: PreparedData, args: argparse.Namespace) -> None:
    stage("4. Compare conservative leakage-safe anchor candidates")
    last_train_origin = int(data.split.val_origins[0]) - HORIZON
    anchor_fit_origins = make_midnight_origins(data.raw, 24 * 14, last_train_origin)
    validation_candidates = fit_anchor_candidates(data.target_raw, anchor_fit_origins)
    anchor_results = evaluate_anchor_candidates(data.target_raw, data.split.val_origins, validation_candidates)
    selected = select_provisional_anchor(anchor_results)
    print_anchor_results(anchor_results, selected)
    print(
        "provisional only: rolling backtest is still required before promoting "
        f"{selected['recipe']} into the PatchTST residual pipeline."
    )

    last_refit_origin = data.split.test_start_idx - HORIZON
    refit_origins = make_midnight_origins(data.raw, 24 * 14, last_refit_origin)
    final_predictor = fit_anchor_candidate(data.target_raw, refit_origins, selected["recipe"])
    test_prediction = make_anchor_predictions(data.target_raw, data.split.test_origins, final_predictor)
    assert_prediction_shapes({"Provisional Anchor": test_prediction}, len(data.split.test_origins))
    save_submission(data.raw, test_prediction, args.output)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "anchor-analysis-only",
        "command": sys.argv,
        "settings": {
            "candidate_policy": "constrained recipes; prefer lowest complexity within 0.5% of holdout minimum MSE",
            "rolling_backtest_required_before_promotion": True,
        },
        "split": {
            "anchor_fit_samples": int(len(anchor_fit_origins)),
            "validation_samples": int(len(data.split.val_origins)),
            "test_samples": int(len(data.split.test_origins)),
            "last_supervised_origin": LAST_ALLOWED_TARGET_START.isoformat(sep=" "),
            "last_supervised_y_end": LAST_ALLOWED_TARGET_END.isoformat(sep=" "),
            "test_input_rule": "each X window ends at its submission timestamp minus one hour",
        },
        "anchor_results": anchor_results,
        "provisional_anchor": {
            **selected,
            "validation_predictor": serialize_anchor_predictor(
                next(candidate for candidate in validation_candidates if candidate["recipe"] == selected["recipe"])
            ),
            "final_refit_predictor": serialize_anchor_predictor(final_predictor),
        },
        "submission": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
        "kaggle_public_mse": None,
    }
    if args.final_tree_blend_search:
        report["submissions"]["final_tree_blend"] = {
            "path": str(final_blend_output.resolve()),
            "sha256": sha256_file(final_blend_output),
        }
    save_json_report(report, args.report_output)


def anchor_rolling_recipes() -> tuple[str, ...]:
    return (
        LEGACY_THREEWAY_ANCHOR_RECIPE,
        "threeway_week2blend_24block",
        "threeway_week2blend_12block_shrink50",
    )


def build_anchor_rolling_folds(data: PreparedData) -> list[dict]:
    """Create chronological folds with a 96-hour label purge before each validation window."""
    final_val_start = int(data.split.val_origins[0])
    fold_spacing = 45 * 24
    validation_days = 111
    folds = []
    for fold_idx, val_start in enumerate(
        [final_val_start - 3 * fold_spacing, final_val_start - 2 * fold_spacing, final_val_start - fold_spacing, final_val_start],
        start=1,
    ):
        val_end = min(val_start + validation_days * 24 - 1, data.split.test_start_idx - HORIZON)
        fit_last_origin = val_start - HORIZON
        fit_origins = make_midnight_origins(data.raw, 24 * 14, fit_last_origin)
        val_origins = make_midnight_origins(data.raw, val_start, val_end)
        assert len(fit_origins) > 0 and len(val_origins) > 0
        assert int(fit_origins[-1]) + HORIZON <= int(val_origins[0])
        assert int(val_origins[-1]) + HORIZON - 1 < data.split.test_start_idx
        folds.append(
            {
                "fold": fold_idx,
                "fit_origins": fit_origins,
                "val_origins": val_origins,
            }
        )
    return folds


def run_anchor_rolling_backtest(data: PreparedData, args: argparse.Namespace) -> None:
    stage("4. Run leakage-safe anchor rolling backtest")
    recipes = anchor_rolling_recipes()
    baseline_recipe = LEGACY_THREEWAY_ANCHOR_RECIPE
    rows = []
    folds = build_anchor_rolling_folds(data)
    for fold in folds:
        fold_rows = {}
        for recipe in recipes:
            predictor = fit_anchor_candidate(data.target_raw, fold["fit_origins"], recipe)
            prediction = make_anchor_predictions(data.target_raw, fold["val_origins"], predictor)
            score = metric(make_supervised_targets(data.target_raw, fold["val_origins"]), prediction)
            fold_rows[recipe] = {
                "fold": int(fold["fold"]),
                "fit_samples": int(len(fold["fit_origins"])),
                "validation_samples": int(len(fold["val_origins"])),
                "validation_start": data.raw.dates[int(fold["val_origins"][0])].isoformat(sep=" "),
                "validation_y_end": data.raw.dates[int(fold["val_origins"][-1]) + HORIZON - 1].isoformat(sep=" "),
                "recipe": recipe,
                "mse": float(score["mse"]),
                "mae": float(score["mae"]),
            }
        baseline_mse = fold_rows[baseline_recipe]["mse"]
        for recipe in recipes:
            row = fold_rows[recipe]
            row["baseline_mse"] = float(baseline_mse)
            row["improvement"] = float(baseline_mse - row["mse"])
            rows.append(row)

    summary = []
    for recipe in recipes:
        recipe_rows = [row for row in rows if row["recipe"] == recipe]
        mse_values = np.asarray([row["mse"] for row in recipe_rows], dtype=np.float64)
        mae_values = np.asarray([row["mae"] for row in recipe_rows], dtype=np.float64)
        improvements = np.asarray([row["improvement"] for row in recipe_rows], dtype=np.float64)
        summary.append(
            {
                "recipe": recipe,
                "folds": int(len(recipe_rows)),
                "mean_mse": float(mse_values.mean()),
                "max_mse": float(mse_values.max()),
                "mean_mae": float(mae_values.mean()),
                "mean_improvement": float(improvements.mean()),
                "positive_folds": int(np.sum(improvements > 0.0)),
            }
        )
    summary.sort(key=lambda row: (row["mean_mse"], row["max_mse"], row["recipe"]))
    baseline_summary = next(row for row in summary if row["recipe"] == baseline_recipe)
    candidates = [row for row in summary if row["recipe"] != baseline_recipe]
    required_positive_folds = max(1, len(folds) - 1)
    eligible = [
        row
        for row in candidates
        if row["mean_mse"] < baseline_summary["mean_mse"]
        and row["positive_folds"] >= required_positive_folds
        and row["max_mse"] <= baseline_summary["max_mse"] * 1.02
    ]
    promoted = min(eligible, key=lambda row: (row["mean_mse"], row["max_mse"], row["recipe"])) if eligible else None

    print("Anchor Rolling Backtest Summary")
    print("-------------------------------")
    print(f"{'recipe':44s} {'mean_mse':>10s} {'max_mse':>10s} {'mean_mae':>10s} {'positive':>10s}")
    for row in summary:
        marker = " <- promoted" if promoted and row["recipe"] == promoted["recipe"] else ""
        print(
            f"{row['recipe']:44s} {row['mean_mse']:10.6f} {row['max_mse']:10.6f} "
            f"{row['mean_mae']:10.6f} {row['positive_folds']:>7d}/{row['folds']}{marker}"
        )
    promotion_passed = promoted is not None
    promoted_recipe = promoted["recipe"] if promoted else baseline_recipe
    print(f"promotion passed: {promotion_passed}; selected recipe: {promoted_recipe}")

    last_refit_origin = data.split.test_start_idx - HORIZON
    refit_origins = make_midnight_origins(data.raw, 24 * 14, last_refit_origin)
    final_predictor = fit_anchor_candidate(data.target_raw, refit_origins, promoted_recipe)
    test_prediction = make_anchor_predictions(data.target_raw, data.split.test_origins, final_predictor)
    assert_prediction_shapes({"Rolling-selected Anchor": test_prediction}, len(data.split.test_origins))
    save_submission(data.raw, test_prediction, args.output)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "anchor-rolling-backtest",
        "command": sys.argv,
        "settings": {
            "recipes": list(recipes),
            "folds": len(folds),
            "fold_spacing_days": 45,
            "validation_days": 111,
            "label_purge_hours": HORIZON,
            "required_positive_folds": required_positive_folds,
            "max_mse_ratio_limit": 1.02,
        },
        "fold_results": rows,
        "summary": summary,
        "promotion": {
            "passed": promotion_passed,
            "baseline_recipe": baseline_recipe,
            "selected_recipe": promoted_recipe,
            "selected_summary": promoted if promoted else baseline_summary,
            "reason": (
                "candidate improved mean MSE, improved at least 3/4 folds, and kept worst-fold MSE within 2% of baseline"
                if promotion_passed
                else "no candidate passed the rolling promotion gate; retained the baseline recipe"
            ),
        },
        "final_refit_predictor": serialize_anchor_predictor(final_predictor),
        "submission": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
        "kaggle_public_mse": None,
    }
    save_json_report(report, args.report_output)


def run_tree_anchor_rolling_backtest(data: PreparedData, args: argparse.Namespace) -> None:
    stage("4. Run leakage-safe LightGBM residual tree-anchor rolling backtest")
    folds = build_anchor_rolling_folds(data)
    fold_results = []
    for fold in folds:
        heuristic_predictor = fit_anchor_predictor(data.target_raw, fold["fit_origins"], "threeway")
        tree_anchor = fit_tree_anchor_model(
            data,
            heuristic_predictor,
            last_origin=int(fold["fit_origins"][-1]),
            residual_lambda=0.0,
            train_step_hours=args.tree_anchor_train_step_hours,
            estimator_count=args.tree_anchor_estimators,
        )
        true = make_supervised_targets(data.target_raw, fold["val_origins"])
        heuristic = make_anchor_predictions(data.target_raw, fold["val_origins"], heuristic_predictor)
        residual = predict_tree_anchor_residual(data, fold["val_origins"], tree_anchor)
        baseline_score = metric(true, heuristic)
        lambda_results = residual_lambda_rows(true, heuristic, residual, TREE_ANCHOR_LAMBDAS)
        fold_results.append(
            {
                "fold": int(fold["fold"]),
                "fit_samples": int(len(fold["fit_origins"])),
                "tree_train_samples": int(len(tree_anchor["train_origins"])),
                "tree_internal_validation_samples": int(len(tree_anchor["internal_val_origins"])),
                "tree_best_iteration": int(tree_anchor["best_iteration"]),
                "validation_samples": int(len(fold["val_origins"])),
                "validation_start": data.raw.dates[int(fold["val_origins"][0])].isoformat(sep=" "),
                "validation_y_end": data.raw.dates[int(fold["val_origins"][-1]) + HORIZON - 1].isoformat(sep=" "),
                "heuristic_mse": float(baseline_score["mse"]),
                "heuristic_mae": float(baseline_score["mae"]),
                "lambda_results": lambda_results,
            }
        )
        best = min(lambda_results, key=lambda row: (row["mse"], row["lambda"]))
        print(
            f"fold={fold['fold']}: heuristic_mse={baseline_score['mse']:.6f}; "
            f"best_lambda={best['lambda']:.2f}; best_mse={best['mse']:.6f}; "
            f"improvement={best['improvement']:+.6f}"
        )

    summary = []
    for residual_lambda in TREE_ANCHOR_LAMBDAS:
        rows = [
            next(row for row in fold["lambda_results"] if np.isclose(row["lambda"], residual_lambda))
            for fold in fold_results
        ]
        baseline_mse = np.asarray([row["baseline_mse"] for row in rows], dtype=np.float64)
        mse_values = np.asarray([row["mse"] for row in rows], dtype=np.float64)
        mae_values = np.asarray([row["mae"] for row in rows], dtype=np.float64)
        improvements = baseline_mse - mse_values
        summary.append(
            {
                "lambda": float(residual_lambda),
                "folds": int(len(rows)),
                "mean_mse": float(mse_values.mean()),
                "max_mse": float(mse_values.max()),
                "mean_mae": float(mae_values.mean()),
                "baseline_mean_mse": float(baseline_mse.mean()),
                "baseline_max_mse": float(baseline_mse.max()),
                "mean_improvement": float(improvements.mean()),
                "positive_folds": int(np.sum(improvements > 0.0)),
            }
        )
    baseline_summary = next(row for row in summary if np.isclose(row["lambda"], 0.0))
    required_positive_folds = max(1, len(folds) - 1)
    eligible = [
        row
        for row in summary
        if row["lambda"] > 0.0
        and row["mean_mse"] < baseline_summary["mean_mse"]
        and row["positive_folds"] >= required_positive_folds
        and row["max_mse"] <= baseline_summary["max_mse"] * 1.02
    ]
    promoted = min(eligible, key=lambda row: (row["mean_mse"], row["max_mse"], row["lambda"])) if eligible else None
    selected_lambda = float(promoted["lambda"]) if promoted else 0.0

    print("Tree-anchor Rolling Backtest Summary")
    print("------------------------------------")
    print(f"{'lambda':>8s} {'mean_mse':>12s} {'max_mse':>12s} {'mean_mae':>12s} {'improvement':>14s} {'positive':>10s}")
    for row in summary:
        marker = " <- promoted" if promoted and np.isclose(row["lambda"], promoted["lambda"]) else ""
        print(
            f"{row['lambda']:8.2f} {row['mean_mse']:12.6f} {row['max_mse']:12.6f} "
            f"{row['mean_mae']:12.6f} {row['mean_improvement']:+14.6f} "
            f"{row['positive_folds']:>7d}/{row['folds']}{marker}"
        )
    print(f"promotion passed: {promoted is not None}; selected lambda: {selected_lambda:.2f}")

    last_refit_origin = data.split.test_start_idx - HORIZON
    refit_origins = make_midnight_origins(data.raw, 24 * 14, last_refit_origin)
    final_heuristic = fit_anchor_predictor(data.target_raw, refit_origins, "threeway")
    if promoted:
        final_tree_anchor = fit_tree_anchor_model(
            data,
            final_heuristic,
            last_origin=last_refit_origin,
            residual_lambda=selected_lambda,
            train_step_hours=args.tree_anchor_train_step_hours,
            estimator_count=args.tree_anchor_estimators,
        )
        test_prediction = make_tree_anchor_predictions(data, data.split.test_origins, final_tree_anchor)
    else:
        final_tree_anchor = None
        test_prediction = make_anchor_predictions(data.target_raw, data.split.test_origins, final_heuristic)
    assert_prediction_shapes({"Rolling-selected Tree Anchor": test_prediction}, len(data.split.test_origins))
    save_submission(data.raw, test_prediction, args.output)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "tree-anchor-rolling-backtest",
        "command": sys.argv,
        "settings": {
            "lambdas": list(TREE_ANCHOR_LAMBDAS),
            "folds": len(folds),
            "train_step_hours": args.tree_anchor_train_step_hours,
            "estimators": args.tree_anchor_estimators,
            "required_positive_folds": required_positive_folds,
            "max_mse_ratio_limit": 1.02,
        },
        "fold_results": fold_results,
        "summary": summary,
        "promotion": {
            "passed": promoted is not None,
            "selected_lambda": selected_lambda,
            "selected_summary": promoted if promoted else baseline_summary,
        },
        "final_tree_anchor": serialize_tree_anchor(final_tree_anchor),
        "submission": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
        "kaggle_public_mse": None,
    }
    save_json_report(report, args.report_output)


def build_patch_residual_rolling_folds(data: PreparedData) -> list[dict]:
    """Nested chronological folds: tune on inner validation, report on unseen outer validation."""
    final_outer_start = int(data.split.val_origins[0])
    fold_spacing = 45 * 24
    validation_days = 45
    folds = []
    for fold_idx, outer_start in enumerate(
        [
            final_outer_start - 3 * fold_spacing,
            final_outer_start - 2 * fold_spacing,
            final_outer_start - fold_spacing,
            final_outer_start,
        ],
        start=1,
    ):
        inner_start = outer_start - validation_days * 24
        inner_end = outer_start - HORIZON
        outer_end = outer_start + validation_days * 24 - 1
        inner_val_origins = make_midnight_origins(data.raw, inner_start, inner_end)
        outer_val_origins = make_midnight_origins(data.raw, outer_start, outer_end)
        last_train_origin = inner_start - HORIZON
        assert len(inner_val_origins) > 0 and len(outer_val_origins) > 0
        assert int(inner_val_origins[-1]) + HORIZON <= int(outer_val_origins[0])
        assert int(outer_val_origins[-1]) + HORIZON - 1 < data.split.test_start_idx
        folds.append(
            {
                "fold": fold_idx,
                "inner_start": inner_start,
                "last_train_origin": last_train_origin,
                "inner_val_origins": inner_val_origins,
                "outer_val_origins": outer_val_origins,
            }
        )
    return folds


def residual_lambda_rows(
    true: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    residual_lambdas: tuple[float, ...],
    block_lambdas: tuple[tuple[float, ...], ...] = (),
) -> list[dict]:
    rows = []
    baseline_score = metric(true, baseline)
    for residual_lambda in residual_lambdas:
        prediction = apply_residual_lambda(baseline, residual, float(residual_lambda))
        score = metric(true, prediction)
        rows.append(
            {
                "lambda_kind": "scalar",
                "lambda": float(residual_lambda),
                "block_lambdas": None,
                "lambda_label": f"{float(residual_lambda):.2f}",
                "mse": float(score["mse"]),
                "mae": float(score["mae"]),
                "baseline_mse": float(baseline_score["mse"]),
                "improvement": float(baseline_score["mse"] - score["mse"]),
            }
        )
    for block_lambda in block_lambdas:
        prediction = apply_residual_lambda(baseline, residual, tuple(float(value) for value in block_lambda))
        score = metric(true, prediction)
        rows.append(
            {
                "lambda_kind": "block",
                "lambda": None,
                "block_lambdas": [float(value) for value in block_lambda],
                "lambda_label": "block[" + ",".join(f"{float(value):.2f}" for value in block_lambda) + "]",
                "mse": float(score["mse"]),
                "mae": float(score["mae"]),
                "baseline_mse": float(baseline_score["mse"]),
                "improvement": float(baseline_score["mse"] - score["mse"]),
            }
        )
    return rows


def final_blend_weight_rows(
    true: np.ndarray,
    base_prediction: np.ndarray,
    correction: np.ndarray,
    weights: tuple[float, ...],
    validation_segments: list[np.ndarray],
) -> list[dict]:
    rows = []
    base_score = metric(true, base_prediction)
    for weight in weights:
        prediction = (base_prediction + float(weight) * correction).astype(np.float32)
        score = metric(true, prediction)
        segment_results = []
        for segment_idx, indices in enumerate(validation_segments, start=1):
            base_segment = metric(true[indices], base_prediction[indices])
            blend_segment = metric(true[indices], prediction[indices])
            segment_results.append(
                {
                    "segment": int(segment_idx),
                    "base_mse": float(base_segment["mse"]),
                    "candidate_mse": float(blend_segment["mse"]),
                    "improvement": float(base_segment["mse"] - blend_segment["mse"]),
                }
            )
        rows.append(
            {
                "weight": float(weight),
                "mse": float(score["mse"]),
                "mae": float(score["mae"]),
                "base_mse": float(base_score["mse"]),
                "base_mae": float(base_score["mae"]),
                "improvement": float(base_score["mse"] - score["mse"]),
                "positive_validation_segments": int(sum(row["improvement"] > 0.0 for row in segment_results)),
                "min_segment_improvement": float(min(row["improvement"] for row in segment_results)),
                "validation_segments": segment_results,
            }
        )
    return rows


def seed_subset_candidate_name(input_size: int, seeds: tuple[int, ...], all_seeds: tuple[int, ...]) -> str:
    base_name = f"PatchTSTResidual-{input_size}"
    if tuple(seeds) == tuple(all_seeds):
        return base_name
    seed_label = "+".join(str(seed) for seed in seeds)
    return f"{base_name}-seeds{seed_label}"


def build_block_lambda_candidates(args: argparse.Namespace) -> tuple[tuple[float, ...], ...]:
    smooth_candidates = (
        smooth_block_lambda_candidates(
            parse_float_list(args.block_lambda_values, "--block-lambda-values"),
            args.block_lambda_max_adjacent_diff,
        )
        if args.block_lambda_search
        else ()
    )
    decay_candidates = (
        parse_block_lambda_candidates(
            args.horizon_decay_lambda_candidates,
            "--horizon-decay-lambda-candidates",
        )
        if args.horizon_decay_lambda_search
        else ()
    )
    return merge_block_lambda_candidates(smooth_candidates, decay_candidates)


def run_patch_residual_rolling_backtest(data: PreparedData, args: argparse.Namespace) -> None:
    require_torch()
    stage("4. Run nested PatchTST residual rolling diagnostics")
    device = device_name()
    patch_lookbacks = parse_int_list(args.patch_lookbacks, "--patch-lookbacks")
    if any(lookback < 24 for lookback in patch_lookbacks):
        raise ValueError("--patch-lookbacks values must be at least 24 hours.")
    residual_lambdas = parse_float_list(args.residual_lambdas, "--residual-lambdas")
    block_lambdas = build_block_lambda_candidates(args)
    folds = build_patch_residual_rolling_folds(data)
    fold_results = []
    training_history = []
    print(
        f"device={device}; seed={args.rolling_patch_seed}; lookbacks={patch_lookbacks}; "
        f"pretrain={args.rolling_pretrain_epochs}; finetune={args.rolling_finetune_epochs}; "
        f"patience={args.rolling_patience}; lambdas={residual_lambdas}; "
        f"block_lambdas={len(block_lambdas)}; window_revin={args.patch_window_revin}; "
        f"lr={args.patch_learning_rate}; weight_decay={args.patch_weight_decay}; dropout={args.patch_dropout}"
    )

    for fold in folds:
        fold_idx = int(fold["fold"])
        inner_val_origins = fold["inner_val_origins"]
        outer_val_origins = fold["outer_val_origins"]
        last_train_origin = int(fold["last_train_origin"])
        anchor_fit_origins = make_midnight_origins(data.raw, 24 * 14, last_train_origin)
        anchor_predictor = fit_anchor_predictor(data.target_raw, anchor_fit_origins, "threeway")
        fold_train_midnight_origins = make_midnight_origins(data.raw, max(patch_lookbacks), last_train_origin)

        fold_split = SplitData(
            test_start_idx=data.split.test_start_idx,
            train_origins=fold_train_midnight_origins,
            val_origins=inner_val_origins,
            test_origins=outer_val_origins,
            scaler_fit_end_idx=int(fold["inner_start"]),
        )
        fold_data = prepare_data(
            data.raw,
            fold_split,
            use_multivariate_input=not args.univariate_input,
            include_raw_calendar=args.include_raw_calendar,
            scaler_fit_end_idx=int(fold["inner_start"]),
        )
        tree_anchor = None
        if args.tree_anchor:
            tree_anchor = fit_tree_anchor_model(
                fold_data,
                anchor_predictor,
                last_origin=last_train_origin,
                residual_lambda=args.tree_anchor_lambda,
                train_step_hours=args.tree_anchor_train_step_hours,
                estimator_count=args.tree_anchor_estimators,
            )
        inner_baseline = make_pipeline_anchor_predictions(fold_data, inner_val_origins, anchor_predictor, tree_anchor)
        outer_baseline = make_pipeline_anchor_predictions(fold_data, outer_val_origins, anchor_predictor, tree_anchor)
        inner_true = make_supervised_targets(data.target_raw, inner_val_origins)
        outer_true = make_supervised_targets(data.target_raw, outer_val_origins)
        candidate_inner_residuals = {}
        candidate_outer_residuals = {}
        fold_histories = []
        lookback_samples = {}
        for input_size in patch_lookbacks:
            train_all_hour_origins = make_all_hour_origins(input_size, last_train_origin)
            train_midnight_origins = make_midnight_origins(data.raw, input_size, last_train_origin)
            train_all_hour_baseline = make_pipeline_anchor_predictions(
                fold_data, train_all_hour_origins, anchor_predictor, tree_anchor
            )
            train_midnight_baseline = make_pipeline_anchor_predictions(
                fold_data, train_midnight_origins, anchor_predictor, tree_anchor
            )
            residual_scale = fit_residual_scale(data.target_raw, train_all_hour_origins, train_all_hour_baseline)
            candidate_name = f"PatchTSTResidual-{input_size}"
            print(
                f"PatchResidualBT{fold_idx}-{input_size}: train_all={len(train_all_hour_origins):,}; "
                f"train_midnight={len(train_midnight_origins):,}; inner={len(inner_val_origins)}; "
                f"outer={len(outer_val_origins)}; residual_scale={residual_scale:.6f}; "
                f"anchor={pipeline_anchor_name(anchor_predictor, tree_anchor)}"
            )
            inner_residual, outer_residual, ema_inner_residual, ema_outer_residual, history = train_patchtst_residual_seed(
                name=f"PatchResidualBT{fold_idx}-{input_size}",
                data=fold_data,
                input_size=input_size,
                seed=args.rolling_patch_seed,
                train_all_hour_origins=train_all_hour_origins,
                train_midnight_origins=train_midnight_origins,
                train_all_hour_baseline=train_all_hour_baseline,
                train_midnight_baseline=train_midnight_baseline,
                val_baseline=inner_baseline,
                residual_scale=residual_scale,
                pretrain_epochs=args.rolling_pretrain_epochs,
                finetune_epochs=args.rolling_finetune_epochs,
                patience=args.rolling_patience,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                learning_rate=args.patch_learning_rate,
                weight_decay=args.patch_weight_decay,
                dropout=args.patch_dropout,
                window_revin=args.patch_window_revin,
                ema_enabled=args.patch_ema,
                ema_decay=args.patch_ema_decay,
            )
            candidate_inner_residuals[candidate_name] = inner_residual
            candidate_outer_residuals[candidate_name] = outer_residual
            if args.patch_ema and ema_inner_residual is not None and ema_outer_residual is not None:
                ema_candidate_name = f"{candidate_name}-ema"
                candidate_inner_residuals[ema_candidate_name] = ema_inner_residual
                candidate_outer_residuals[ema_candidate_name] = ema_outer_residual
            lookback_samples[candidate_name] = {
                "train_all_hour_samples": int(len(train_all_hour_origins)),
                "train_midnight_samples": int(len(train_midnight_origins)),
                "residual_scale": float(residual_scale),
            }
            fold_histories.append({"fold": fold_idx, "candidate": candidate_name, **history})

        if len(patch_lookbacks) > 1:
            blend_name = "PatchTSTResidual-blend-" + "-".join(str(value) for value in patch_lookbacks)
            candidate_inner_residuals[blend_name] = np.mean(
                [candidate_inner_residuals[f"PatchTSTResidual-{value}"] for value in patch_lookbacks],
                axis=0,
            )
            candidate_outer_residuals[blend_name] = np.mean(
                [candidate_outer_residuals[f"PatchTSTResidual-{value}"] for value in patch_lookbacks],
                axis=0,
            )
            if args.patch_ema:
                ema_blend_name = f"{blend_name}-ema"
                candidate_inner_residuals[ema_blend_name] = np.mean(
                    [candidate_inner_residuals[f"PatchTSTResidual-{value}-ema"] for value in patch_lookbacks],
                    axis=0,
                )
                candidate_outer_residuals[ema_blend_name] = np.mean(
                    [candidate_outer_residuals[f"PatchTSTResidual-{value}-ema"] for value in patch_lookbacks],
                    axis=0,
                )

        candidate_results = {}
        for candidate_name, inner_residual in candidate_inner_residuals.items():
            outer_residual = candidate_outer_residuals[candidate_name]
            inner_rows = residual_lambda_rows(inner_true, inner_baseline, inner_residual, residual_lambdas, block_lambdas)
            outer_rows = residual_lambda_rows(outer_true, outer_baseline, outer_residual, residual_lambdas, block_lambdas)
            inner_best = min(inner_rows, key=lambda row: (row["mse"], lambda_complexity(row), lambda_label(row)))
            outer_by_label = {lambda_label(row): row for row in outer_rows}
            selected_outer = outer_by_label[lambda_label(inner_best)]
            candidate_results[candidate_name] = {
                "inner_lambda_results": inner_rows,
                "outer_lambda_results": outer_rows,
                "inner_selected_lambda": lambda_values_for_report(lambda_spec_from_row(inner_best)),
                "outer_selected_mse": float(selected_outer["mse"]),
                "outer_selected_improvement": float(selected_outer["improvement"]),
            }
        fold_results.append(
            {
                "fold": fold_idx,
                "inner_validation_start": data.raw.dates[int(inner_val_origins[0])].isoformat(sep=" "),
                "inner_validation_y_end": data.raw.dates[int(inner_val_origins[-1]) + HORIZON - 1].isoformat(sep=" "),
                "outer_validation_start": data.raw.dates[int(outer_val_origins[0])].isoformat(sep=" "),
                "outer_validation_y_end": data.raw.dates[int(outer_val_origins[-1]) + HORIZON - 1].isoformat(sep=" "),
                "lookback_samples": lookback_samples,
                "tree_anchor": serialize_tree_anchor(tree_anchor),
                "candidate_results": candidate_results,
            }
        )
        training_history.extend(fold_histories)

    lambda_summary = []
    candidate_names = list(fold_results[0]["candidate_results"])
    for candidate_name in candidate_names:
        labels = [lambda_label(row) for row in fold_results[0]["candidate_results"][candidate_name]["outer_lambda_results"]]
        for label in labels:
            rows = [
                next(
                    row
                    for row in fold["candidate_results"][candidate_name]["outer_lambda_results"]
                    if lambda_label(row) == label
                )
                for fold in fold_results
            ]
            improvements = np.asarray([row["improvement"] for row in rows], dtype=np.float64)
            mse_values = np.asarray([row["mse"] for row in rows], dtype=np.float64)
            baseline_values = np.asarray([row["baseline_mse"] for row in rows], dtype=np.float64)
            lambda_summary.append(
                {
                    "candidate": candidate_name,
                    "lambda_kind": rows[0]["lambda_kind"],
                    "lambda": rows[0]["lambda"],
                    "block_lambdas": rows[0]["block_lambdas"],
                    "lambda_label": lambda_label(rows[0]),
                    "mean_mse": float(mse_values.mean()),
                    "max_mse": float(mse_values.max()),
                    "baseline_mean_mse": float(baseline_values.mean()),
                    "baseline_max_mse": float(baseline_values.max()),
                    "mean_improvement": float(improvements.mean()),
                    "positive_folds": int(np.sum(improvements > 0.0)),
                    "folds": int(len(rows)),
                }
            )
    candidate_diagnostics = {}
    for candidate_name in candidate_names:
        candidate_summary = [row for row in lambda_summary if row["candidate"] == candidate_name]
        selected_outer_improvements = np.asarray(
            [fold["candidate_results"][candidate_name]["outer_selected_improvement"] for fold in fold_results],
            dtype=np.float64,
        )
        candidate_diagnostics[candidate_name] = {
            "fixed_lambda_passes": [
                row
                for row in candidate_summary
                if lambda_is_active(row)
                and row["mean_improvement"] > 0.0
                and row["positive_folds"] >= len(folds) - 1
                and row["max_mse"] <= row["baseline_max_mse"] * 1.02
            ],
            "inner_selected_positive_outer_folds": int(np.sum(selected_outer_improvements > 0.0)),
            "inner_selected_mean_outer_improvement": float(selected_outer_improvements.mean()),
        }
    diagnostic = {
        "candidate_diagnostics": candidate_diagnostics,
    }
    print("PatchTST Residual Rolling Summary")
    print("---------------------------------")
    print(f"{'candidate':36s} {'lambda':>32s} {'mean_mse':>12s} {'max_mse':>12s} {'mean_improvement':>18s} {'positive':>10s}")
    for row in lambda_summary:
        print(
            f"{row['candidate']:36s} {row['lambda_label']:>32s} {row['mean_mse']:12.6f} {row['max_mse']:12.6f} "
            f"{row['mean_improvement']:18.6f} {row['positive_folds']:>7d}/{row['folds']}"
        )
    print(f"diagnostic: {diagnostic}")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "patch-residual-rolling-backtest",
        "command": sys.argv,
        "settings": {
            "device": device,
            "seed": args.rolling_patch_seed,
            "patch_lookbacks": list(patch_lookbacks),
            "pretrain_epochs": args.rolling_pretrain_epochs,
            "finetune_epochs": args.rolling_finetune_epochs,
            "patience": args.rolling_patience,
            "residual_lambdas": list(residual_lambdas),
            "block_lambda_search": bool(args.block_lambda_search),
            "block_lambda_values": list(parse_float_list(args.block_lambda_values, "--block-lambda-values")),
            "block_lambda_max_adjacent_diff": float(args.block_lambda_max_adjacent_diff),
            "horizon_decay_lambda_search": bool(args.horizon_decay_lambda_search),
            "horizon_decay_lambda_candidates": [
                list(candidate)
                for candidate in parse_block_lambda_candidates(
                    args.horizon_decay_lambda_candidates,
                    "--horizon-decay-lambda-candidates",
                )
            ],
            "block_lambda_candidate_count": int(len(block_lambdas)),
            "nested_validation": True,
            "label_purge_hours": HORIZON,
            "window_revin": bool(args.patch_window_revin),
            "patch_learning_rate": float(args.patch_learning_rate),
            "patch_weight_decay": float(args.patch_weight_decay),
            "patch_dropout": float(args.patch_dropout),
            "patch_ema": bool(args.patch_ema),
            "patch_ema_decay": float(args.patch_ema_decay),
            "tree_anchor": bool(args.tree_anchor),
            "tree_anchor_lambda": float(args.tree_anchor_lambda),
            "tree_anchor_train_step_hours": int(args.tree_anchor_train_step_hours),
            "tree_anchor_estimators": int(args.tree_anchor_estimators),
        },
        "fold_results": fold_results,
        "lambda_summary": lambda_summary,
        "diagnostic": diagnostic,
        "training_history": training_history,
    }
    save_json_report(report, args.report_output)


def run_patchtst_residual_pipeline(data: PreparedData, args: argparse.Namespace) -> None:
    require_torch()
    stage("4. Train leakage-safe anchor + PatchTST residual candidates")
    device = device_name()
    patch_seeds = parse_seed_list(args.patch_seeds)
    patch_lookbacks = parse_int_list(args.patch_lookbacks, "--patch-lookbacks")
    residual_lambdas = parse_float_list(args.residual_lambdas, "--residual-lambdas")
    block_lambdas = build_block_lambda_candidates(args)
    pretrain_epochs = 1 if args.quick else args.pretrain_epochs
    finetune_epochs = 1 if args.quick else args.finetune_epochs
    patience = 1 if args.quick else args.patience
    if args.quick:
        patch_seeds = patch_seeds[:1]
    if any(lookback < 24 for lookback in patch_lookbacks):
        raise ValueError("--patch-lookbacks values must be at least 24 hours.")
    if any(value < 0.0 for value in residual_lambdas):
        raise ValueError("--residual-lambdas values must be non-negative.")
    if not any(np.isclose(value, 0.0) for value in residual_lambdas):
        raise ValueError("--residual-lambdas must include 0 for a leakage-safe anchor fallback.")
    if args.validation_segments < 1:
        raise ValueError("--validation-segments must be at least 1.")
    if args.validation_segments > len(data.split.val_origins):
        raise ValueError("--validation-segments cannot exceed the number of validation origins.")
    if not 1 <= args.min_positive_validation_segments <= args.validation_segments:
        raise ValueError("--min-positive-validation-segments must be between 1 and --validation-segments.")

    print(f"torch device: {device}")
    print(f"PatchTST residual lookbacks: {patch_lookbacks}")
    print(f"PatchTST residual seeds: {patch_seeds}")
    print(f"anchor strategy: {args.anchor}")
    print(f"PatchTST RevIN-style window normalization: {args.patch_window_revin}")
    print(
        f"PatchTST optimizer/model regularization: lr={args.patch_learning_rate}; "
        f"weight_decay={args.patch_weight_decay}; dropout={args.patch_dropout}"
    )
    print(f"pretrain epochs: {pretrain_epochs}; finetune epochs: {finetune_epochs}; patience: {patience}")
    print(f"lambda grid: {residual_lambdas}")
    print(f"block lambda search: {args.block_lambda_search}; candidates={len(block_lambdas)}")
    print(
        f"validation stability gate: at least {args.min_positive_validation_segments}/"
        f"{args.validation_segments} chronological segments must beat the anchor"
    )

    last_train_origin = int(data.split.val_origins[0]) - HORIZON
    anchor_fit_origins = make_midnight_origins(data.raw, 24 * 14, last_train_origin)
    validation_anchor_predictor, validation_tree_anchor = fit_pipeline_anchor(
        data,
        anchor_fit_origins,
        last_origin=last_train_origin,
        args=args,
        quick=args.quick,
    )
    val_true = make_supervised_targets(data.target_raw, data.split.val_origins)
    val_baseline = make_pipeline_anchor_predictions(
        data,
        data.split.val_origins,
        validation_anchor_predictor,
        validation_tree_anchor,
    )
    validation_stage_test_baseline = make_pipeline_anchor_predictions(
        data,
        data.split.test_origins,
        validation_anchor_predictor,
        validation_tree_anchor,
    )
    print(
        f"validation anchor: {pipeline_anchor_name(validation_anchor_predictor, validation_tree_anchor)}; "
        f"fit samples={len(anchor_fit_origins):,}; MSE={metric(val_true, val_baseline)['mse']:.6f}"
    )
    validation_residuals: dict[str, np.ndarray] = {}
    validation_stage_test_residuals: dict[str, np.ndarray] = {}
    candidate_lookbacks: dict[str, tuple[int, ...]] = {}
    candidate_seed_map: dict[str, dict[int, tuple[int, ...]]] = {}
    candidate_use_ema: dict[str, bool] = {}
    training_history: dict[str, list[dict]] = {}

    for input_size in patch_lookbacks:
        train_all_hour_origins = make_all_hour_origins(input_size, last_train_origin)
        train_midnight_origins = make_midnight_origins(data.raw, input_size, last_train_origin)
        train_all_hour_baseline = make_pipeline_anchor_predictions(
            data,
            train_all_hour_origins,
            validation_anchor_predictor,
            validation_tree_anchor,
        )
        train_midnight_baseline = make_pipeline_anchor_predictions(
            data,
            train_midnight_origins,
            validation_anchor_predictor,
            validation_tree_anchor,
        )
        residual_scale = fit_residual_scale(data.target_raw, train_all_hour_origins, train_all_hour_baseline)
        print(
            f"PatchTSTResidual-{input_size}: all-hour pretrain samples={len(train_all_hour_origins):,}, "
            f"midnight finetune samples={len(train_midnight_origins):,}, residual_scale={residual_scale:.6f}"
        )
        seed_validation_residuals = []
        seed_test_residuals = []
        seed_ema_validation_residuals = []
        seed_ema_test_residuals = []
        seed_validation_residual_by_seed = {}
        seed_test_residual_by_seed = {}
        seed_ema_validation_residual_by_seed = {}
        seed_ema_test_residual_by_seed = {}
        histories = []
        for seed in patch_seeds:
            val_residual, test_residual, ema_val_residual, ema_test_residual, history = train_patchtst_residual_seed(
                name=f"PatchTSTResidual-{input_size}",
                data=data,
                input_size=input_size,
                seed=seed,
                train_all_hour_origins=train_all_hour_origins,
                train_midnight_origins=train_midnight_origins,
                train_all_hour_baseline=train_all_hour_baseline,
                train_midnight_baseline=train_midnight_baseline,
                val_baseline=val_baseline,
                residual_scale=residual_scale,
                pretrain_epochs=pretrain_epochs,
                finetune_epochs=finetune_epochs,
                patience=patience,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                learning_rate=args.patch_learning_rate,
                weight_decay=args.patch_weight_decay,
                dropout=args.patch_dropout,
                window_revin=args.patch_window_revin,
                ema_enabled=args.patch_ema,
                ema_decay=args.patch_ema_decay,
            )
            seed_validation_residuals.append(val_residual)
            seed_test_residuals.append(test_residual)
            seed_validation_residual_by_seed[int(seed)] = val_residual
            seed_test_residual_by_seed[int(seed)] = test_residual
            if args.patch_ema and ema_val_residual is not None and ema_test_residual is not None:
                seed_ema_validation_residuals.append(ema_val_residual)
                seed_ema_test_residuals.append(ema_test_residual)
                seed_ema_validation_residual_by_seed[int(seed)] = ema_val_residual
                seed_ema_test_residual_by_seed[int(seed)] = ema_test_residual
            histories.append(history)

        seed_subsets: list[tuple[int, ...]] = [tuple(patch_seeds)]
        if args.patch_seed_candidate_search and len(patch_seeds) > 1:
            seed_subsets.extend(
                subset
                for subset_size in range(1, len(patch_seeds))
                for subset in itertools.combinations(tuple(patch_seeds), subset_size)
            )
        seen_seed_subsets = set()
        for seed_subset in seed_subsets:
            seed_subset = tuple(int(seed) for seed in seed_subset)
            if seed_subset in seen_seed_subsets:
                continue
            seen_seed_subsets.add(seed_subset)
            candidate_name = seed_subset_candidate_name(input_size, seed_subset, tuple(patch_seeds))
            validation_residuals[candidate_name] = np.mean(
                [seed_validation_residual_by_seed[seed] for seed in seed_subset],
                axis=0,
            )
            validation_stage_test_residuals[candidate_name] = np.mean(
                [seed_test_residual_by_seed[seed] for seed in seed_subset],
                axis=0,
            )
            candidate_lookbacks[candidate_name] = (input_size,)
            candidate_seed_map[candidate_name] = {int(input_size): seed_subset}
            candidate_use_ema[candidate_name] = False
            if args.patch_ema:
                ema_candidate_name = f"{candidate_name}-ema"
                validation_residuals[ema_candidate_name] = np.mean(
                    [seed_ema_validation_residual_by_seed[seed] for seed in seed_subset],
                    axis=0,
                )
                validation_stage_test_residuals[ema_candidate_name] = np.mean(
                    [seed_ema_test_residual_by_seed[seed] for seed in seed_subset],
                    axis=0,
                )
                candidate_lookbacks[ema_candidate_name] = (input_size,)
                candidate_seed_map[ema_candidate_name] = {int(input_size): seed_subset}
                candidate_use_ema[ema_candidate_name] = True
        training_history[f"PatchTSTResidual-{input_size}"] = histories

    if len(patch_lookbacks) > 1:
        blend_name = "PatchTSTResidual-blend-" + "-".join(str(value) for value in patch_lookbacks)
        validation_residuals[blend_name] = np.mean(
            [validation_residuals[f"PatchTSTResidual-{value}"] for value in patch_lookbacks],
            axis=0,
        )
        validation_stage_test_residuals[blend_name] = np.mean(
            [validation_stage_test_residuals[f"PatchTSTResidual-{value}"] for value in patch_lookbacks],
            axis=0,
        )
        candidate_lookbacks[blend_name] = patch_lookbacks
        candidate_seed_map[blend_name] = {int(value): tuple(patch_seeds) for value in patch_lookbacks}
        candidate_use_ema[blend_name] = False
        if args.patch_ema:
            ema_blend_name = f"{blend_name}-ema"
            validation_residuals[ema_blend_name] = np.mean(
                [validation_residuals[f"PatchTSTResidual-{value}-ema"] for value in patch_lookbacks],
                axis=0,
            )
            validation_stage_test_residuals[ema_blend_name] = np.mean(
                [validation_stage_test_residuals[f"PatchTSTResidual-{value}-ema"] for value in patch_lookbacks],
                axis=0,
            )
            candidate_lookbacks[ema_blend_name] = patch_lookbacks
            candidate_seed_map[ema_blend_name] = {int(value): tuple(patch_seeds) for value in patch_lookbacks}
            candidate_use_ema[ema_blend_name] = True

    stage("5. Select the PatchTST residual candidate and lambda on validation")
    lambda_search = []
    eligible_candidates = []
    validation_segments = np.array_split(np.arange(len(data.split.val_origins)), args.validation_segments)
    for candidate_name, residual in validation_residuals.items():
        lambda_specs: list[float | tuple[float, ...]] = [float(value) for value in residual_lambdas]
        lambda_specs.extend(block_lambdas)
        for residual_lambda in lambda_specs:
            prediction = apply_residual_lambda(val_baseline, residual, residual_lambda)
            candidate_metric = metric(val_true, prediction)
            segment_results = []
            for segment_idx, indices in enumerate(validation_segments, start=1):
                baseline_metric = metric(val_true[indices], val_baseline[indices])
                segment_metric = metric(val_true[indices], prediction[indices])
                segment_results.append(
                    {
                        "segment": segment_idx,
                        "baseline_mse": float(baseline_metric["mse"]),
                        "candidate_mse": float(segment_metric["mse"]),
                        "improvement": float(baseline_metric["mse"] - segment_metric["mse"]),
                    }
                )
            positive_segments = sum(row["improvement"] > 0.0 for row in segment_results)
            min_segment_improvement = min(row["improvement"] for row in segment_results)
            seed_count = sum(
                len(seeds)
                for seeds in candidate_seed_map.get(
                    candidate_name,
                    {int(input_size): tuple(patch_seeds) for input_size in candidate_lookbacks[candidate_name]},
                ).values()
            )
            row = {
                "candidate": candidate_name,
                **lambda_values_for_report(residual_lambda),
                **candidate_metric,
                "positive_validation_segments": int(positive_segments),
                "min_segment_improvement": float(min_segment_improvement),
                "seed_count": int(seed_count),
                "validation_segments": segment_results,
            }
            lambda_search.append(row)
            if (
                lambda_is_active(row)
                and positive_segments >= args.min_positive_validation_segments
                and row["mse"] < metric(val_true, val_baseline)["mse"]
            ):
                eligible_candidates.append(row)
    if eligible_candidates:
        best_mse = min(float(row["mse"]) for row in eligible_candidates)
        mse_limit = best_mse * (1.0 + max(0.0, float(args.selection_mse_tolerance)))
        near_best_candidates = [row for row in eligible_candidates if float(row["mse"]) <= mse_limit]
        stable_near_best_candidates = [
            row
            for row in near_best_candidates
            if float(row["min_segment_improvement"]) >= float(args.selection_min_segment_improvement_floor)
        ]
        if stable_near_best_candidates:
            best_candidate = min(
                stable_near_best_candidates,
                key=lambda row: (
                    -int(row["seed_count"]),
                    -float(row["min_segment_improvement"]),
                    float(row["mse"]),
                    lambda_complexity(row),
                    lambda_label(row),
                    str(row["candidate"]),
                ),
            )
        else:
            best_candidate = min(
                near_best_candidates,
                key=lambda row: (
                    float(row["mse"]),
                    lambda_complexity(row),
                    lambda_label(row),
                    -float(row["min_segment_improvement"]),
                    str(row["candidate"]),
                ),
            )
    else:
        fallback_name = next(iter(validation_residuals))
        best_candidate = next(
            row
            for row in lambda_search
            if row["candidate"] == fallback_name
            and row["lambda_kind"] == "scalar"
            and np.isclose(float(row["lambda"]), 0.0)
        )
        print("No stable residual correction passed the validation segment gate. Falling back to the anchor.")

    selected_name = str(best_candidate["candidate"])
    selected_lambda = lambda_spec_from_row(best_candidate)
    selected_validation_prediction = apply_residual_lambda(
        val_baseline,
        validation_residuals[selected_name],
        selected_lambda,
    )
    final_tree_blend_validation_model = None
    final_tree_blend_rows: list[dict] = []
    final_tree_blend_best = {
        "weight": 0.0,
        "mse": float(best_candidate["mse"]),
        "mae": float(best_candidate["mae"]),
        "improvement": 0.0,
        "positive_validation_segments": 0,
        "min_segment_improvement": 0.0,
        "validation_segments": [],
    }
    final_tree_blend_weight = 0.0
    if args.final_tree_blend_search:
        stage("5b. Search a small final LightGBM residual blend on validation")
        final_tree_blend_validation_model = fit_tree_anchor_model(
            data,
            validation_anchor_predictor,
            last_origin=last_train_origin,
            residual_lambda=0.0,
            train_step_hours=args.tree_anchor_train_step_hours,
            estimator_count=min(args.tree_anchor_estimators, 120) if args.quick else args.tree_anchor_estimators,
        )
        final_tree_blend_residual = predict_tree_anchor_residual(
            data,
            data.split.val_origins,
            final_tree_blend_validation_model,
        )
        final_tree_blend_rows = final_blend_weight_rows(
            val_true,
            selected_validation_prediction,
            final_tree_blend_residual,
            parse_float_list(args.final_tree_blend_weights, "--final-tree-blend-weights"),
            list(validation_segments),
        )
        eligible_tree_blends = [
            row
            for row in final_tree_blend_rows
            if abs(float(row["weight"])) > 1e-12
            and float(row["improvement"]) > 0.0
            and int(row["positive_validation_segments"]) >= args.final_tree_blend_min_positive_segments
        ]
        if eligible_tree_blends:
            final_tree_blend_best = min(
                eligible_tree_blends,
                key=lambda row: (float(row["mse"]), abs(float(row["weight"])), -float(row["min_segment_improvement"])),
            )
            final_tree_blend_weight = float(final_tree_blend_best["weight"])
        else:
            final_tree_blend_best = next(row for row in final_tree_blend_rows if np.isclose(float(row["weight"]), 0.0))
            final_tree_blend_weight = 0.0
            print("No final LightGBM residual blend passed the validation gate. Keeping PatchTST prediction.")
        print(
            "Final LightGBM blend: "
            f"weight={final_tree_blend_weight:.3f}; "
            f"MSE={final_tree_blend_best['mse']:.6f}; MAE={final_tree_blend_best['mae']:.6f}; "
            f"segments={final_tree_blend_best['positive_validation_segments']}/{args.validation_segments}"
        )
    anchor_label = f"Anchor: {pipeline_anchor_name(validation_anchor_predictor, validation_tree_anchor)}"
    validation_results = {
        "Seasonal Naive": metric(val_true, make_seasonal_naive_predictions(data.target_raw, data.split.val_origins)),
        anchor_label: metric(val_true, val_baseline),
    }
    for candidate_name, residual in validation_residuals.items():
        validation_results[f"{candidate_name} lambda=1.00"] = metric(val_true, val_baseline + residual)
    validation_results["Selected PatchTST Residual"] = {
        "mse": float(best_candidate["mse"]),
        "mae": float(best_candidate["mae"]),
    }
    if args.final_tree_blend_search:
        validation_results["Selected PatchTST + final LightGBM residual blend"] = {
            "mse": float(final_tree_blend_best["mse"]),
            "mae": float(final_tree_blend_best["mae"]),
        }
    print("Validation Results")
    print("------------------")
    for name, values in validation_results.items():
        print(f"{name}: MSE = {values['mse']:.6f}, MAE = {values['mae']:.6f}")
    print(f"Best Overall: {selected_name}, lambda = {lambda_label(best_candidate)}")
    print(
        "Validation segment gate: "
        f"{best_candidate['positive_validation_segments']}/{args.validation_segments} positive segments"
    )

    final_refit_history: dict[str, list[dict]] = {}
    final_anchor_predictor = validation_anchor_predictor
    final_tree_anchor = validation_tree_anchor
    final_tree_blend_refit_model = None
    final_test_baseline = validation_stage_test_baseline
    if args.no_final_refit:
        final_refit_scaler_end = data.split.scaler_fit_end_idx
    else:
        last_refit_origin = data.split.test_start_idx - HORIZON
        refit_anchor_origins = make_midnight_origins(data.raw, 24 * 14, last_refit_origin)
        final_anchor_predictor, final_tree_anchor = fit_pipeline_anchor(
            data,
            refit_anchor_origins,
            last_origin=last_refit_origin,
            args=args,
            quick=args.quick,
        )
        final_test_baseline = make_pipeline_anchor_predictions(
            data,
            data.split.test_origins,
            final_anchor_predictor,
            final_tree_anchor,
        )
        final_refit_scaler_end = data.split.test_start_idx
        print(
            f"final anchor refit: {pipeline_anchor_name(final_anchor_predictor, final_tree_anchor)}; "
            f"fit samples={len(refit_anchor_origins):,}"
        )

    if not lambda_is_active(best_candidate):
        print("Selected lambda is 0.00. PatchTST refit is unnecessary because the submission equals the anchor.")
        selected_test_residual = np.zeros_like(final_test_baseline)
    elif args.no_final_refit:
        print("warning: --no-final-refit keeps the validation-stage model. Do not use this mode for the final submission.")
        selected_test_residual = validation_stage_test_residuals[selected_name]
    else:
        stage("6. Refit the selected PatchTST residual model on all allowed pre-test labels")
        refit_data = prepare_data(
            data.raw,
            data.split,
            use_multivariate_input=not args.univariate_input,
            include_raw_calendar=args.include_raw_calendar,
            scaler_fit_end_idx=data.split.test_start_idx,
        )
        selected_lookbacks = candidate_lookbacks[selected_name]
        selected_seed_map = candidate_seed_map.get(
            selected_name,
            {int(input_size): tuple(patch_seeds) for input_size in selected_lookbacks},
        )
        selected_uses_ema = bool(candidate_use_ema.get(selected_name, False))
        refit_test_residuals = []
        for input_size in selected_lookbacks:
            last_refit_origin = data.split.test_start_idx - HORIZON
            refit_all_hour_origins = make_all_hour_origins(input_size, last_refit_origin)
            refit_midnight_origins = make_midnight_origins(data.raw, input_size, last_refit_origin)
            refit_all_hour_baseline = make_pipeline_anchor_predictions(
                refit_data,
                refit_all_hour_origins,
                final_anchor_predictor,
                final_tree_anchor,
            )
            refit_midnight_baseline = make_pipeline_anchor_predictions(
                refit_data,
                refit_midnight_origins,
                final_anchor_predictor,
                final_tree_anchor,
            )
            residual_scale = fit_residual_scale(
                refit_data.target_raw,
                refit_all_hour_origins,
                refit_all_hour_baseline,
            )
            print(
                f"PatchTSTResidual-{input_size} final refit: all-hour samples={len(refit_all_hour_origins):,}, "
                f"midnight samples={len(refit_midnight_origins):,}, residual_scale={residual_scale:.6f}"
            )
            lookback_test_residuals = []
            histories = []
            validation_histories = training_history[f"PatchTSTResidual-{input_size}"]
            validation_history_by_seed = {int(history["seed"]): history for history in validation_histories}
            selected_seeds = tuple(int(seed) for seed in selected_seed_map.get(int(input_size), tuple(patch_seeds)))
            print(f"PatchTSTResidual-{input_size} selected refit seeds: {selected_seeds}")
            for seed in selected_seeds:
                validation_history = validation_history_by_seed[seed]
                if selected_uses_ema and validation_history.get("best_ema_finetune_epoch"):
                    selected_finetune_epochs = max(1, int(validation_history["best_ema_finetune_epoch"]))
                else:
                    selected_finetune_epochs = max(1, int(validation_history["best_finetune_epoch"]))
                test_residual, ema_test_residual, history = fit_patchtst_residual_refit_seed(
                    name=f"PatchTSTResidual-{input_size}",
                    data=refit_data,
                    input_size=input_size,
                    seed=seed,
                    train_all_hour_origins=refit_all_hour_origins,
                    train_midnight_origins=refit_midnight_origins,
                    train_all_hour_baseline=refit_all_hour_baseline,
                    train_midnight_baseline=refit_midnight_baseline,
                    residual_scale=residual_scale,
                    pretrain_epochs=pretrain_epochs,
                    finetune_epochs=selected_finetune_epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    learning_rate=args.patch_learning_rate,
                    weight_decay=args.patch_weight_decay,
                    dropout=args.patch_dropout,
                    window_revin=args.patch_window_revin,
                    ema_enabled=selected_uses_ema,
                    ema_decay=args.patch_ema_decay,
                )
                lookback_test_residuals.append(
                    ema_test_residual if selected_uses_ema and ema_test_residual is not None else test_residual
                )
                histories.append(history)
            refit_test_residuals.append(np.mean(lookback_test_residuals, axis=0))
            final_refit_history[f"PatchTSTResidual-{input_size}"] = histories
        selected_test_residual = np.mean(refit_test_residuals, axis=0)

    stage("7. Verify predictions and write submit.csv")
    patch_only_prediction = apply_residual_lambda(final_test_baseline, selected_test_residual, selected_lambda)
    final_prediction = patch_only_prediction
    if args.final_tree_blend_search and abs(final_tree_blend_weight) > 1e-12:
        last_refit_origin = data.split.test_start_idx - HORIZON
        final_tree_blend_refit_model = fit_tree_anchor_model(
            data,
            final_anchor_predictor,
            last_origin=last_refit_origin,
            residual_lambda=0.0,
            train_step_hours=args.tree_anchor_train_step_hours,
            estimator_count=min(args.tree_anchor_estimators, 120) if args.quick else args.tree_anchor_estimators,
        )
        final_tree_blend_test_residual = predict_tree_anchor_residual(
            data,
            data.split.test_origins,
            final_tree_blend_refit_model,
        )
        final_prediction = (patch_only_prediction + final_tree_blend_weight * final_tree_blend_test_residual).astype(
            np.float32
        )
    seasonal_naive_test_prediction = make_seasonal_naive_predictions(data.target_raw, data.split.test_origins)
    assert_prediction_shapes(
        {
            "Seasonal Naive": seasonal_naive_test_prediction,
            "Anchor": final_test_baseline,
            "Selected PatchTST Residual": patch_only_prediction,
            "Final Prediction": final_prediction,
        },
        len(data.split.test_origins),
    )
    save_submission(data.raw, final_prediction, args.output)
    seasonal_output = derived_submission_path(args.output, "seasonal_naive")
    anchor_output = derived_submission_path(args.output, "anchor")
    patch_output = derived_submission_path(args.output, "patchtst_residual")
    final_blend_output = derived_submission_path(args.output, "final_tree_blend")
    save_submission(data.raw, seasonal_naive_test_prediction, seasonal_output)
    save_submission(data.raw, final_test_baseline, anchor_output)
    save_submission(data.raw, patch_only_prediction, patch_output)
    if args.final_tree_blend_search:
        save_submission(data.raw, final_prediction, final_blend_output)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "patchtst-residual",
        "command": sys.argv,
        "settings": {
            "input_mode": "multivariate + cyclic time features" if not args.univariate_input else "univariate OT only",
            "include_raw_calendar": bool(args.include_raw_calendar),
            "anchor": args.anchor,
            "tree_anchor": bool(args.tree_anchor),
            "tree_anchor_lambda": float(args.tree_anchor_lambda),
            "tree_anchor_train_step_hours": int(args.tree_anchor_train_step_hours),
            "tree_anchor_estimators": int(args.tree_anchor_estimators),
            "final_tree_blend_search": bool(args.final_tree_blend_search),
            "final_tree_blend_weights": list(parse_float_list(args.final_tree_blend_weights, "--final-tree-blend-weights")),
            "final_tree_blend_min_positive_segments": int(args.final_tree_blend_min_positive_segments),
            "block_lambda_search": bool(args.block_lambda_search),
            "block_lambda_values": list(parse_float_list(args.block_lambda_values, "--block-lambda-values")),
            "block_lambda_max_adjacent_diff": float(args.block_lambda_max_adjacent_diff),
            "horizon_decay_lambda_search": bool(args.horizon_decay_lambda_search),
            "horizon_decay_lambda_candidates": [
                list(candidate)
                for candidate in parse_block_lambda_candidates(
                    args.horizon_decay_lambda_candidates,
                    "--horizon-decay-lambda-candidates",
                )
            ],
            "block_lambda_candidate_count": int(len(block_lambdas)),
            "patch_window_revin": bool(args.patch_window_revin),
            "patch_learning_rate": float(args.patch_learning_rate),
            "patch_weight_decay": float(args.patch_weight_decay),
            "patch_dropout": float(args.patch_dropout),
            "patch_ema": bool(args.patch_ema),
            "patch_ema_decay": float(args.patch_ema_decay),
            "patch_seed_candidate_search": bool(args.patch_seed_candidate_search),
            "patch_seeds": list(patch_seeds),
            "patch_lookbacks": list(patch_lookbacks),
            "pretrain_epochs": pretrain_epochs,
            "finetune_epochs": finetune_epochs,
            "patience": patience,
            "validation_segments": args.validation_segments,
            "min_positive_validation_segments": args.min_positive_validation_segments,
            "selection_mse_tolerance": float(args.selection_mse_tolerance),
            "selection_min_segment_improvement_floor": float(args.selection_min_segment_improvement_floor),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "quick": bool(args.quick),
            "final_refit": not args.no_final_refit,
        },
        "split": {
            "validation_train_samples": int(len(data.split.train_origins)),
            "validation_samples": int(len(data.split.val_origins)),
            "test_samples": int(len(data.split.test_origins)),
            "last_supervised_origin": LAST_ALLOWED_TARGET_START.isoformat(sep=" "),
            "last_supervised_y_end": LAST_ALLOWED_TARGET_END.isoformat(sep=" "),
            "validation_scaler_fit_end_exclusive": data.raw.dates[data.split.scaler_fit_end_idx].isoformat(sep=" "),
            "final_refit_scaler_fit_end_exclusive": data.raw.dates[final_refit_scaler_end].isoformat(sep=" "),
            "test_input_rule": "each X window ends at its submission timestamp minus one hour",
        },
        "validation_results": validation_results,
        "lambda_search": sorted(lambda_search, key=lambda row: row["mse"]),
        "final_tree_blend": {
            "enabled": bool(args.final_tree_blend_search),
            "selected_weight": float(final_tree_blend_weight),
            "selected_result": final_tree_blend_best,
            "weight_search": sorted(final_tree_blend_rows, key=lambda row: row["mse"]),
            "validation_model": serialize_tree_anchor(final_tree_blend_validation_model),
            "final_refit_model": serialize_tree_anchor(final_tree_blend_refit_model),
        },
        "validation_anchor_predictor": serialize_anchor_predictor(validation_anchor_predictor),
        "validation_tree_anchor": serialize_tree_anchor(validation_tree_anchor),
        "final_anchor_predictor": serialize_anchor_predictor(final_anchor_predictor),
        "final_tree_anchor": serialize_tree_anchor(final_tree_anchor),
        "selected_predictor": {
            "name": selected_name,
            **lambda_values_for_report(selected_lambda),
            "mse": float(best_candidate["mse"]),
            "mae": float(best_candidate["mae"]),
            "lookbacks": list(candidate_lookbacks[selected_name]),
            "uses_ema": bool(candidate_use_ema.get(selected_name, False)),
            "seeds_by_lookback": {
                str(input_size): list(seeds)
                for input_size, seeds in candidate_seed_map.get(
                    selected_name,
                    {int(input_size): tuple(patch_seeds) for input_size in candidate_lookbacks[selected_name]},
                ).items()
            },
        },
        "final_predictor": {
            "name": (
                f"{selected_name} + final_lightgbm_residual_blend"
                if args.final_tree_blend_search and abs(final_tree_blend_weight) > 1e-12
                else selected_name
            ),
            "patchtst_candidate": selected_name,
            **lambda_values_for_report(selected_lambda),
            "patchtst_uses_ema": bool(candidate_use_ema.get(selected_name, False)),
            "final_tree_blend_weight": float(final_tree_blend_weight),
            "validation_mse": float(final_tree_blend_best["mse"] if args.final_tree_blend_search else best_candidate["mse"]),
            "validation_mae": float(final_tree_blend_best["mae"] if args.final_tree_blend_search else best_candidate["mae"]),
        },
        "training_history": training_history,
        "final_refit_history": final_refit_history,
        "submissions": {
            "selected": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
            "seasonal_naive": {"path": str(seasonal_output.resolve()), "sha256": sha256_file(seasonal_output)},
            "anchor": {"path": str(anchor_output.resolve()), "sha256": sha256_file(anchor_output)},
            "patchtst_residual": {"path": str(patch_output.resolve()), "sha256": sha256_file(patch_output)},
        },
        "kaggle_public_mse": None,
    }
    save_json_report(report, args.report_output)
    print("Report Summary")
    print("--------------")
    print("Base deep learning model: PatchTST residual forecaster")
    print(f"Anchor: {pipeline_anchor_name(final_anchor_predictor, final_tree_anchor)}")
    print(f"Selected predictor: {selected_name}, lambda={lambda_label(best_candidate)}")
    print(f"Final refit enabled: {not args.no_final_refit}")


def run_validate_only(data: PreparedData, output_path: Path) -> None:
    stage("4. Validate-only seasonal-naive smoke run")
    val_true = make_supervised_targets(data.target_raw, data.split.val_origins)
    val_pred = make_seasonal_naive_predictions(data.target_raw, data.split.val_origins)
    test_pred = make_seasonal_naive_predictions(data.target_raw, data.split.test_origins)
    assert_prediction_shapes({"Seasonal Naive": test_pred}, len(data.split.test_origins))
    result = {"Seasonal Naive": metric(val_true, val_pred)}
    print_validation_results(result, best_single="Seasonal Naive", best_overall="Seasonal Naive")
    save_submission(data.raw, test_pred, output_path)
    print_summary(
        used_models=["Seasonal Naive (--validate-only)"],
        best_single="Seasonal Naive",
        best_overall="Seasonal Naive",
        selected_weights={"Seasonal Naive": 1.0},
    )


def run_full_pipeline(data: PreparedData, args: argparse.Namespace) -> None:
    require_torch()
    stage("4. Train PyTorch forecasting models")
    device = device_name()
    epochs = 1 if args.quick else args.epochs
    patience = 1 if args.quick else args.patience
    patch_seeds = parse_seed_list(args.patch_seeds)
    if args.quick:
        patch_seeds = patch_seeds[:1]
    num_features = len(data.feature_names)
    print(f"torch device: {device}")
    print(f"epochs: {epochs}; patience: {patience}; patch seeds: {patch_seeds}")

    patch_512_val, patch_512_test, patch_512_history = train_seed_family(
        name="PatchTST-512",
        builder=lambda: PatchTST(input_size=512, num_features=num_features),
        data=data,
        input_size=512,
        seeds=patch_seeds,
        learning_rate=3e-4,
        epochs=epochs,
        patience=patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    patch_336_val, patch_336_test, patch_336_history = train_seed_family(
        name="PatchTST-336",
        builder=lambda: PatchTST(input_size=336, num_features=num_features),
        data=data,
        input_size=336,
        seeds=patch_seeds,
        learning_rate=3e-4,
        epochs=epochs,
        patience=patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    dlinear_val, dlinear_test, dlinear_history = train_seed_family(
        name="DLinear",
        builder=lambda: DLinear(input_size=DLINEAR_LOOKBACK, num_features=num_features),
        data=data,
        input_size=DLINEAR_LOOKBACK,
        seeds=(42,),
        learning_rate=1e-3,
        epochs=epochs,
        patience=patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    nbeats_val, nbeats_test, nbeats_history = train_seed_family(
        name="N-BEATS",
        builder=lambda: NBeats(input_size=NBEATS_LOOKBACK, num_features=num_features),
        data=data,
        input_size=NBEATS_LOOKBACK,
        seeds=(42,),
        learning_rate=1e-3,
        epochs=epochs,
        patience=patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    nhits_val, nhits_test, nhits_history = train_seed_family(
        name="N-HiTS",
        builder=lambda: NHiTS(input_size=NHITS_LOOKBACK, num_features=num_features),
        data=data,
        input_size=NHITS_LOOKBACK,
        seeds=(42,),
        learning_rate=1e-3,
        epochs=epochs,
        patience=patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    validation_predictions = {
        "PatchTST": 0.6 * patch_512_val + 0.4 * patch_336_val,
        "DLinear/NLinear": dlinear_val,
        "N-BEATS": nbeats_val,
        "N-HiTS": nhits_val,
        "Seasonal Naive": make_seasonal_naive_predictions(data.target_raw, data.split.val_origins),
    }
    test_predictions = {
        "PatchTST": 0.6 * patch_512_test + 0.4 * patch_336_test,
        "DLinear/NLinear": dlinear_test,
        "N-BEATS": nbeats_test,
        "N-HiTS": nhits_test,
        "Seasonal Naive": make_seasonal_naive_predictions(data.target_raw, data.split.test_origins),
    }
    training_history = {
        "PatchTST-512": patch_512_history,
        "PatchTST-336": patch_336_history,
        "DLinear": dlinear_history,
        "N-BEATS": nbeats_history,
        "N-HiTS": nhits_history,
    }

    if not args.skip_gbm:
        gbm_predictions = train_optional_gbm(data, quick=args.quick)
        if gbm_predictions is not None:
            validation_predictions["GBM optional"], test_predictions["GBM optional"] = gbm_predictions

    stage("6. Evaluate fixed and inverse-MSE ensembles")
    val_true = make_supervised_targets(data.target_raw, data.split.val_origins)
    fixed_val = weighted_prediction(validation_predictions, FIXED_WEIGHTS)
    fixed_test = weighted_prediction(test_predictions, FIXED_WEIGHTS)
    inverse_members = {name: validation_predictions[name] for name in FIXED_WEIGHTS}
    inverse_weights = inverse_mse_weights(inverse_members, val_true)
    inverse_val = weighted_prediction(validation_predictions, inverse_weights)
    inverse_test = weighted_prediction(test_predictions, inverse_weights)

    ensemble_validation_predictions = {
        "Fixed Ensemble": fixed_val,
        "Inverse-MSE Weighted Ensemble": inverse_val,
    }
    ensemble_test_predictions = {
        "Fixed Ensemble": fixed_test,
        "Inverse-MSE Weighted Ensemble": inverse_test,
    }
    ensemble_weights = {
        "Fixed Ensemble": FIXED_WEIGHTS,
        "Inverse-MSE Weighted Ensemble": inverse_weights,
    }

    if "GBM optional" in validation_predictions:
        fixed_gbm_val = weighted_prediction(validation_predictions, FIXED_WEIGHTS_WITH_GBM)
        fixed_gbm_test = weighted_prediction(test_predictions, FIXED_WEIGHTS_WITH_GBM)
        inverse_gbm_weights = inverse_mse_weights(validation_predictions, val_true)
        inverse_gbm_val = weighted_prediction(validation_predictions, inverse_gbm_weights)
        inverse_gbm_test = weighted_prediction(test_predictions, inverse_gbm_weights)
        ensemble_validation_predictions["Fixed Ensemble with GBM"] = fixed_gbm_val
        ensemble_validation_predictions["Inverse-MSE Weighted Ensemble with GBM"] = inverse_gbm_val
        ensemble_test_predictions["Fixed Ensemble with GBM"] = fixed_gbm_test
        ensemble_test_predictions["Inverse-MSE Weighted Ensemble with GBM"] = inverse_gbm_test
        ensemble_weights["Fixed Ensemble with GBM"] = FIXED_WEIGHTS_WITH_GBM
        ensemble_weights["Inverse-MSE Weighted Ensemble with GBM"] = inverse_gbm_weights

    all_validation_predictions = {**validation_predictions, **ensemble_validation_predictions}
    all_test_predictions = {**test_predictions, **ensemble_test_predictions}
    results = {name: metric(val_true, prediction) for name, prediction in all_validation_predictions.items()}
    single_names = list(validation_predictions)
    best_single = min(single_names, key=lambda name: results[name]["mse"])
    best_overall = min(results, key=lambda name: results[name]["mse"])
    selected_weights = ensemble_weights.get(best_overall, {best_overall: 1.0})

    print(f"fixed ensemble weights: {FIXED_WEIGHTS}")
    print(f"inverse-MSE capped weights: {inverse_weights}")
    if "Fixed Ensemble with GBM" in ensemble_weights:
        print(f"fixed ensemble with GBM weights: {FIXED_WEIGHTS_WITH_GBM}")
        print(f"inverse-MSE capped weights with GBM: {ensemble_weights['Inverse-MSE Weighted Ensemble with GBM']}")
    print_validation_results(results, best_single=best_single, best_overall=best_overall)

    stage("7. Verify predictions and write submit.csv")
    assert_prediction_shapes(test_predictions, len(data.split.test_origins))
    assert_prediction_shapes(ensemble_test_predictions, len(data.split.test_origins))
    save_submission(data.raw, all_test_predictions[best_overall], args.output)
    print_summary(
        used_models=list(validation_predictions),
        best_single=best_single,
        best_overall=best_overall,
        selected_weights=selected_weights,
    )
    save_run_report(
        args=args,
        data=data,
        mode="full",
        results=results,
        best_single=best_single,
        best_overall=best_overall,
        selected_weights=selected_weights,
        used_models=list(validation_predictions),
        training_history=training_history,
    )


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("--patch-seeds must contain at least one integer.")
    return seeds


def main() -> None:
    args = parse_args()
    raw = load_data(args.data, args.sample)
    split = build_splits(raw, validation_fraction=args.validation_fraction)
    use_multivariate_input = USE_MULTIVARIATE_INPUT and not args.univariate_input
    data = prepare_data(
        raw,
        split,
        use_multivariate_input=use_multivariate_input,
        include_raw_calendar=args.include_raw_calendar or args.legacy_multimodel,
    )
    if args.patch_residual_rolling_backtest:
        run_patch_residual_rolling_backtest(data, args)
    elif args.tree_anchor_rolling_backtest:
        run_tree_anchor_rolling_backtest(data, args)
    elif args.anchor_rolling_backtest:
        run_anchor_rolling_backtest(data, args)
    elif args.anchor_analysis_only:
        run_anchor_analysis_only(data, args)
    elif args.validate_only:
        run_validate_only(data, args.output)
        val_true = make_supervised_targets(data.target_raw, data.split.val_origins)
        results = {"Seasonal Naive": metric(val_true, make_seasonal_naive_predictions(data.target_raw, data.split.val_origins))}
        save_run_report(
            args=args,
            data=data,
            mode="validate-only",
            results=results,
            best_single="Seasonal Naive",
            best_overall="Seasonal Naive",
            selected_weights={"Seasonal Naive": 1.0},
            used_models=["Seasonal Naive (--validate-only)"],
        )
    elif args.legacy_multimodel:
        run_full_pipeline(data, args)
    else:
        run_patchtst_residual_pipeline(data, args)


if __name__ == "__main__":
    main()
