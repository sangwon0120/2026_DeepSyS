"""
ETTh1 96-hour OT forecasting project.

This script follows the requested process:
1. problem definition
2. data acquisition
3. preprocessing
4. feature engineering
5. model selection
6. model training
7. model optimization and submission generation

The local environment for this workspace does not include pandas or PyTorch, so
the runnable implementation uses only numpy and the Python standard library.
The model is a direct multi-step MLP trained with Adam from scratch. It mirrors
the tutorial flow: sliding windows, Dataset-like arrays, MSE training, validation
tracking, and horizon-wise metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


RAW_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_COL = "OT"
HORIZON = 96
TEST_START = datetime(2018, 2, 1, 0, 0, 0)


@dataclass
class DataBundle:
    dates: np.ndarray
    date_objects: list[datetime]
    values: np.ndarray
    value_cols: list[str]
    sample_ids: list[str]
    sample_datetimes: np.ndarray
    sample_columns: list[str]
    sample_id_col: str


@dataclass
class FeatureBundle:
    features_raw: np.ndarray
    features_scaled: np.ndarray
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    target_mean: float
    target_std: float
    target_min: float
    target_max: float


@dataclass
class Boundaries:
    test_start_idx: int
    train_boundary_idx: int
    train_starts: np.ndarray
    val_starts: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stage(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def report_checks(title: str, checks: list[tuple[str, bool]]) -> None:
    print("\nVerification checklist:")
    failed = []
    for text, ok in checks:
        mark = "OK" if ok else "FAIL"
        print(f" - [{mark}] {text}")
        if not ok:
            failed.append(text)
    if failed:
        raise ValueError(f"{title} failed checks: {failed}")
    print("Decision: checks passed; proceed to the next step.")


def parse_datetime(value: str) -> datetime:
    value = value.strip()
    if len(value) == 10:
        return datetime.strptime(value, "%Y-%m-%d")
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def to_datetime64_hour(dt: datetime) -> np.datetime64:
    return np.datetime64(dt.replace(minute=0, second=0, microsecond=0), "h")


def load_data(data_path: Path, sample_path: Path) -> DataBundle:
    date_objects: list[datetime] = []
    values: list[list[float]] = []

    with data_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in ["date", *RAW_FEATURES] if c not in fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {data_path}: {missing}")

        for row_idx, row in enumerate(reader, start=2):
            try:
                date_objects.append(parse_datetime(row["date"]))
                values.append([float(row[c]) for c in RAW_FEATURES])
            except Exception as exc:
                raise ValueError(f"Failed to parse row {row_idx} in {data_path}: {exc}") from exc

    sample_ids: list[str] = []
    sample_datetimes: list[np.datetime64] = []
    with sample_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sample_columns = reader.fieldnames or []
        if "ID" in sample_columns:
            sample_id_col = "ID"
        elif "timestamp" in sample_columns:
            sample_id_col = "timestamp"
        else:
            raise ValueError("sample_submit.csv must contain ID or timestamp column.")
        for row in reader:
            sample_ids.append(row[sample_id_col])
            sample_datetimes.append(to_datetime64_hour(parse_datetime(row[sample_id_col])))

    dates = np.array([to_datetime64_hour(dt) for dt in date_objects], dtype="datetime64[h]")
    return DataBundle(
        dates=dates,
        date_objects=date_objects,
        values=np.asarray(values, dtype=np.float32),
        value_cols=RAW_FEATURES.copy(),
        sample_ids=sample_ids,
        sample_datetimes=np.asarray(sample_datetimes, dtype="datetime64[h]"),
        sample_columns=sample_columns,
        sample_id_col=sample_id_col,
    )


def define_problem(bundle: DataBundle) -> None:
    stage("1. Problem definition")
    t_cols = [c for c in bundle.sample_columns if c.startswith("T")]
    expected_t_cols = [f"T{i}" for i in range(HORIZON)]
    print(f"Target: {TARGET_COL} / horizon: T0..T{HORIZON - 1} ({HORIZON} hours)")
    print(f"Evaluation metric: MSE on the 96 predicted OT values")
    print(f"Training/validation rows must be earlier than {TEST_START:%Y-%m-%d %H:%M:%S}")
    print(f"Submission id column: {bundle.sample_id_col}")
    print(f"Submission columns: {bundle.sample_columns[:5]} ... {bundle.sample_columns[-3:]}")
    report_checks(
        "Problem definition",
        [
            ("train/validation/test boundary is explicitly tied to 2018-02-01 00:00:00", True),
            ("submission has exactly T0..T95 output columns", t_cols == expected_t_cols),
            ("model output must be a 96-dimensional vector", HORIZON == 96),
            ("data leakage risk is documented and checked in code", True),
        ],
    )


def acquire_and_validate_data(bundle: DataBundle) -> None:
    stage("2. Data acquisition")
    diffs = np.diff(bundle.dates).astype("timedelta64[h]").astype(int)
    t_cols = [c for c in bundle.sample_columns if c.startswith("T")]
    print(f"ETTh1 shape: rows={bundle.values.shape[0]}, columns={len(bundle.value_cols) + 1}")
    print(f"ETTh1 columns: ['date', {', '.join(bundle.value_cols)}]")
    print(f"ETTh1 date range: {bundle.dates[0]} ~ {bundle.dates[-1]}")
    print(f"Sample submission rows: {len(bundle.sample_ids)}, columns={len(bundle.sample_columns)}")
    print(f"Sample ID range: {bundle.sample_ids[0]} ~ {bundle.sample_ids[-1]}")
    print(f"Missing numeric values: {int(np.isnan(bundle.values).sum())}")
    print(f"Unique hour diffs: {np.unique(diffs)[:10].tolist()}")
    report_checks(
        "Data acquisition",
        [
            ("hourly interval is continuous", bool(np.all(diffs == 1))),
            ("all required raw columns exist", bundle.value_cols == RAW_FEATURES),
            ("sample submission contains 96 target columns", len(t_cols) == HORIZON),
            ("sample submission ID count is positive", len(bundle.sample_ids) > 0),
            ("no missing numeric values are present", not bool(np.isnan(bundle.values).any())),
        ],
    )


def get_time_boundaries(
    dates: np.ndarray,
    sample_datetimes: np.ndarray,
    train_fraction_before_test: float,
) -> tuple[int, int]:
    matches = np.where(dates == to_datetime64_hour(TEST_START))[0]
    if len(matches) != 1:
        raise ValueError(f"Expected one row at test start {TEST_START}, found {len(matches)}")
    test_start_idx = int(matches[0])

    # The project describes train/val/test as 60/20/20. Since the test boundary
    # is fixed at 2018-02-01, using 0.75 of the pre-test period gives roughly
    # 60% train and 20% validation over the full timeline.
    train_boundary_idx = int(test_start_idx * train_fraction_before_test)
    train_boundary_idx = max(0, (train_boundary_idx // 24) * 24)

    missing_ids = [str(x) for x in sample_datetimes if x not in dates]
    if missing_ids:
        raise ValueError(f"Sample IDs not found in ETTh1 date index: {missing_ids[:5]}")
    return test_start_idx, train_boundary_idx


def preprocess_data(bundle: DataBundle, train_fraction_before_test: float) -> tuple[int, int]:
    stage("3. Data preprocessing")
    test_start_idx, train_boundary_idx = get_time_boundaries(
        bundle.dates, bundle.sample_datetimes, train_fraction_before_test
    )
    train_start = bundle.dates[0]
    train_end = bundle.dates[train_boundary_idx - 1]
    val_start = bundle.dates[train_boundary_idx]
    val_end = bundle.dates[test_start_idx - 1]
    test_start = bundle.dates[test_start_idx]
    print(f"Train rows for scaler/model target period: [0, {train_boundary_idx})")
    print(f"Validation target rows: [{train_boundary_idx}, {test_start_idx})")
    print(f"Test/submission period starts at row {test_start_idx}: {test_start}")
    print(f"Train date range: {train_start} ~ {train_end}")
    print(f"Validation date range: {val_start} ~ {val_end}")
    print(f"Numeric dtype: {bundle.values.dtype}")
    report_checks(
        "Data preprocessing",
        [
            ("validation period is later than train period", train_boundary_idx < test_start_idx),
            ("all train/validation rows are before 2018-02-01", bundle.dates[test_start_idx] == to_datetime64_hour(TEST_START)),
            ("2018-02-01 and later rows are excluded from training and validation targets", bool(np.all(bundle.dates[:test_start_idx] < to_datetime64_hour(TEST_START)))),
            ("train boundary is aligned to midnight", bundle.date_objects[train_boundary_idx].hour == 0),
        ],
    )
    return test_start_idx, train_boundary_idx


def add_time_features(bundle: DataBundle) -> tuple[np.ndarray, list[str]]:
    time_rows: list[list[float]] = []
    for dt in bundle.date_objects:
        hour = dt.hour
        dow = dt.weekday()
        month = dt.month - 1
        doy = dt.timetuple().tm_yday - 1
        time_rows.append(
            [
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                math.sin(2 * math.pi * dow / 7),
                math.cos(2 * math.pi * dow / 7),
                math.sin(2 * math.pi * month / 12),
                math.cos(2 * math.pi * month / 12),
                math.sin(2 * math.pi * doy / 366),
                math.cos(2 * math.pi * doy / 366),
            ]
        )
    time_names = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
    ]
    features = np.concatenate([bundle.values, np.asarray(time_rows, dtype=np.float32)], axis=1)
    return features.astype(np.float32), [*bundle.value_cols, *time_names]


def fit_transform_scaler(
    features_raw: np.ndarray,
    feature_names: list[str],
    train_boundary_idx: int,
) -> FeatureBundle:
    train_slice = features_raw[:train_boundary_idx]
    mean = train_slice.mean(axis=0)
    std = train_slice.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    features_scaled = (features_raw - mean) / std
    target_idx = feature_names.index(TARGET_COL)
    target_values_train = features_raw[:train_boundary_idx, target_idx]
    return FeatureBundle(
        features_raw=features_raw,
        features_scaled=features_scaled.astype(np.float32),
        feature_names=feature_names,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        target_mean=float(mean[target_idx]),
        target_std=float(std[target_idx]),
        target_min=float(target_values_train.min()),
        target_max=float(target_values_train.max()),
    )


def make_window_starts(
    test_start_idx: int,
    train_boundary_idx: int,
    lookback: int,
    horizon: int,
) -> Boundaries:
    train_stop = train_boundary_idx - lookback - horizon + 1
    val_start = train_boundary_idx - lookback
    val_stop = test_start_idx - lookback - horizon + 1
    if train_stop <= 0:
        raise ValueError("Not enough training rows for the requested lookback/horizon.")
    if val_stop <= val_start:
        raise ValueError("Not enough validation rows for the requested lookback/horizon.")
    return Boundaries(
        test_start_idx=test_start_idx,
        train_boundary_idx=train_boundary_idx,
        train_starts=np.arange(0, train_stop, dtype=np.int64),
        val_starts=np.arange(val_start, val_stop, dtype=np.int64),
    )


def make_windows(
    features_scaled: np.ndarray,
    ot_scaled: np.ndarray,
    starts: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_idx = starts[:, None] + np.arange(lookback, dtype=np.int64)[None, :]
    y_idx = starts[:, None] + lookback + np.arange(horizon, dtype=np.int64)[None, :]
    x = features_scaled[x_idx].reshape(len(starts), lookback * features_scaled.shape[1])
    y = ot_scaled[y_idx]
    return x.astype(np.float32), y.astype(np.float32)


def feature_engineering(
    bundle: DataBundle,
    test_start_idx: int,
    train_boundary_idx: int,
    lookbacks: list[int],
) -> FeatureBundle:
    stage("4. Feature engineering")
    features_raw, feature_names = add_time_features(bundle)
    features = fit_transform_scaler(features_raw, feature_names, train_boundary_idx)
    print(f"Base raw features: {RAW_FEATURES}")
    print(f"Engineered feature names: {feature_names}")
    print(f"Feature matrix shape: {features.features_scaled.shape}")
    print(f"Scaler fit rows: [0, {train_boundary_idx}) only")
    print(f"OT scaler mean/std: {features.target_mean:.4f} / {features.target_std:.4f}")

    min_l = min(lookbacks)
    boundaries = make_window_starts(test_start_idx, train_boundary_idx, min_l, HORIZON)
    first_start = int(boundaries.train_starts[0])
    first_x = (first_start, first_start + min_l - 1)
    first_y = (first_start + min_l, first_start + min_l + HORIZON - 1)
    print(f"Example with L={min_l}: X rows {first_x}, y rows {first_y}")

    report_checks(
        "Feature engineering",
        [
            ("feature count D matches engineered matrix", features.features_scaled.shape[1] == len(feature_names)),
            ("target column OT is present in features", TARGET_COL in feature_names),
            ("scaler was fit before validation/test rows", train_boundary_idx < test_start_idx),
            ("first sample target starts exactly after input window", first_x[1] + 1 == first_y[0]),
            ("validation windows can be generated without using test targets", len(boundaries.val_starts) > 0),
        ],
    )
    return features


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def metrics(pred_raw: np.ndarray, true_raw: np.ndarray) -> dict[str, Any]:
    err = pred_raw - true_raw
    horizon_mse = np.mean(err**2, axis=0)
    horizon_rmse = np.sqrt(horizon_mse)
    return {
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "horizon_rmse": horizon_rmse.astype(float).tolist(),
    }


def baseline_predict(
    raw_ot: np.ndarray,
    starts: np.ndarray,
    lookback: int,
    horizon: int,
    kind: str,
) -> np.ndarray:
    if kind == "last_value":
        return np.repeat(raw_ot[starts + lookback - 1, None], horizon, axis=1)
    if kind == "last_96h_repeat":
        if lookback < horizon:
            raise ValueError("last_96h_repeat baseline requires lookback >= horizon")
        idx = starts[:, None] + lookback - horizon + np.arange(horizon, dtype=np.int64)[None, :]
        return raw_ot[idx]
    raise ValueError(f"Unknown baseline kind: {kind}")


class MLPRegressor:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float,
        dropout: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(seed)

        lim1 = math.sqrt(6.0 / (input_dim + hidden_dim))
        lim2 = math.sqrt(6.0 / (hidden_dim + output_dim))
        self.params: dict[str, np.ndarray] = {
            "W1": self.rng.uniform(-lim1, lim1, size=(input_dim, hidden_dim)).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "W2": self.rng.uniform(-lim2, lim2, size=(hidden_dim, output_dim)).astype(np.float32),
            "b2": np.zeros(output_dim, dtype=np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.params = {k: v.copy() for k, v in state.items()}

    def _forward(self, x: np.ndarray, training: bool) -> tuple[np.ndarray, dict[str, np.ndarray | None]]:
        z1 = x @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(z1, 0.0)
        mask = None
        a_used = a1
        if training and self.dropout > 0:
            keep = 1.0 - self.dropout
            mask = (self.rng.random(a1.shape) < keep).astype(np.float32) / keep
            a_used = a1 * mask
        out = a_used @ self.params["W2"] + self.params["b2"]
        cache: dict[str, np.ndarray | None] = {"x": x, "z1": z1, "a1": a1, "a_used": a_used, "mask": mask}
        return out, cache

    def train_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        pred, cache = self._forward(x, training=True)
        diff = pred - y
        loss = float(np.mean(diff**2))
        scale = 2.0 / float(x.shape[0] * y.shape[1])
        d_out = diff * scale

        a_used = cache["a_used"]
        z1 = cache["z1"]
        mask = cache["mask"]
        assert isinstance(a_used, np.ndarray)
        assert isinstance(z1, np.ndarray)

        grads: dict[str, np.ndarray] = {
            "W2": a_used.T @ d_out + self.weight_decay * self.params["W2"],
            "b2": d_out.sum(axis=0),
        }
        d_a1 = d_out @ self.params["W2"].T
        if mask is not None:
            d_a1 *= mask
        d_z1 = d_a1 * (z1 > 0)
        grads["W1"] = x.T @ d_z1 + self.weight_decay * self.params["W1"]
        grads["b1"] = d_z1.sum(axis=0)

        self._adam_step(grads)
        return loss

    def _adam_step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        for name, grad in grads.items():
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - beta1**self.t)
            v_hat = self.v[name] / (1.0 - beta2**self.t)
            self.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        preds = []
        for start in range(0, len(x), batch_size):
            pred, _ = self._forward(x[start : start + batch_size], training=False)
            preds.append(pred)
        return np.concatenate(preds, axis=0).astype(np.float32)


def inverse_target(y_scaled: np.ndarray, features: FeatureBundle) -> np.ndarray:
    return y_scaled * features.target_std + features.target_mean


def train_model(
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    features: FeatureBundle,
    seed: int,
) -> tuple[MLPRegressor, list[dict[str, float]], dict[str, Any]]:
    model = MLPRegressor(
        input_dim=x_train.shape[1],
        hidden_dim=int(config["hidden"]),
        output_dim=HORIZON,
        lr=float(config["lr"]),
        dropout=float(config["dropout"]),
        weight_decay=float(config["weight_decay"]),
        seed=seed,
    )
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    patience = int(config["patience"])
    rng = np.random.default_rng(seed)

    best_state: dict[str, np.ndarray] | None = None
    best_val_mse = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(len(x_train))
        batch_losses = []
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            loss = model.train_batch(x_train[idx], y_train[idx])
            batch_losses.append(loss)

        val_pred_scaled = model.predict(x_val)
        val_pred_raw = inverse_target(val_pred_scaled, features)
        val_true_raw = inverse_target(y_val, features)
        val_metrics = metrics(val_pred_raw, val_true_raw)
        train_loss = float(np.mean(batch_losses))
        row = {
            "epoch": float(epoch),
            "train_loss_scaled": train_loss,
            "val_mse_raw": val_metrics["mse"],
            "val_rmse_raw": val_metrics["rmse"],
            "val_mae_raw": val_metrics["mae"],
        }
        history.append(row)
        print(
            f"[{config['name']}] epoch {epoch:02d} "
            f"train_scaled_mse={train_loss:.5f} "
            f"val_mse={val_metrics['mse']:.5f} "
            f"val_rmse={val_metrics['rmse']:.5f}"
        )

        if val_metrics["mse"] < best_val_mse - 1e-8:
            best_val_mse = val_metrics["mse"]
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{config['name']}] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training produced no best state.")
    model.load_state_dict(best_state)
    best_pred_scaled = model.predict(x_val)
    best_metric = metrics(inverse_target(best_pred_scaled, features), inverse_target(y_val, features))
    return model, history, best_metric


def config_grid(full: bool, epochs_override: int | None) -> list[dict[str, Any]]:
    if full:
        configs = [
            {"name": "mlp_L96_h64", "L": 96, "hidden": 64, "lr": 1e-3, "dropout": 0.05},
            {"name": "mlp_L168_h96", "L": 168, "hidden": 96, "lr": 8e-4, "dropout": 0.08},
            {"name": "mlp_L336_h128", "L": 336, "hidden": 128, "lr": 6e-4, "dropout": 0.10},
        ]
        default_epochs = 25
        patience = 5
    else:
        configs = [
            {"name": "mlp_L96_h64", "L": 96, "hidden": 64, "lr": 1e-3, "dropout": 0.05},
            {"name": "mlp_L168_h64", "L": 168, "hidden": 64, "lr": 8e-4, "dropout": 0.08},
        ]
        default_epochs = 10
        patience = 3

    for cfg in configs:
        cfg["epochs"] = epochs_override if epochs_override is not None else default_epochs
        cfg["batch_size"] = 256
        cfg["weight_decay"] = 1e-5
        cfg["patience"] = patience
    return configs


def model_selection(configs: list[dict[str, Any]]) -> None:
    stage("5. Model selection")
    print("Baselines:")
    print(" - last_value: repeat the last observed OT value for all 96 horizons")
    print(" - last_96h_repeat: repeat the previous 96 observed OT values")
    print("Neural model:")
    print(" - direct multi-step MLP: flattened L x D input -> hidden ReLU -> 96 outputs")
    print("Experiment grid:")
    for cfg in configs:
        print(
            f" - {cfg['name']}: L={cfg['L']}, hidden={cfg['hidden']}, "
            f"lr={cfg['lr']}, dropout={cfg['dropout']}, epochs={cfg['epochs']}"
        )
    report_checks(
        "Model selection",
        [
            ("baseline models are defined before neural training", True),
            ("every neural model outputs batch x 96", all(int(c["hidden"]) > 0 for c in configs)),
            ("experiments can be compared by validation MSE", len(configs) > 0),
        ],
    )


def evaluate_baselines(
    raw_ot: np.ndarray,
    starts: np.ndarray,
    lookback: int,
    true_raw: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for kind in ["last_value", "last_96h_repeat"]:
        pred = baseline_predict(raw_ot, starts, lookback, HORIZON, kind)
        met = metrics(pred, true_raw)
        rows.append(
            {
                "name": kind,
                "kind": "baseline",
                "L": lookback,
                "hidden": "",
                "epochs": 0,
                "mse": met["mse"],
                "rmse": met["rmse"],
                "mae": met["mae"],
                "horizon_rmse": met["horizon_rmse"],
            }
        )
        print(f"[baseline:{kind}] val_mse={met['mse']:.5f} val_rmse={met['rmse']:.5f}")
    return rows


def run_training_and_optimization(
    bundle: DataBundle,
    features: FeatureBundle,
    test_start_idx: int,
    train_boundary_idx: int,
    configs: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, MLPRegressor], dict[str, list[dict[str, float]]]]:
    stage("6. Model training")
    raw_ot = features.features_raw[:, features.feature_names.index(TARGET_COL)]
    ot_scaled = features.features_scaled[:, features.feature_names.index(TARGET_COL)]
    results: list[dict[str, Any]] = []
    trained_models: dict[str, MLPRegressor] = {}
    histories: dict[str, list[dict[str, float]]] = {}

    for cfg_idx, cfg in enumerate(configs):
        lookback = int(cfg["L"])
        boundaries = make_window_starts(test_start_idx, train_boundary_idx, lookback, HORIZON)
        print(
            f"\nPreparing windows for {cfg['name']}: "
            f"train={len(boundaries.train_starts)}, val={len(boundaries.val_starts)}, "
            f"L={lookback}, H={HORIZON}, D={features.features_scaled.shape[1]}"
        )
        x_train, y_train = make_windows(
            features.features_scaled, ot_scaled, boundaries.train_starts, lookback, HORIZON
        )
        x_val, y_val = make_windows(
            features.features_scaled, ot_scaled, boundaries.val_starts, lookback, HORIZON
        )
        true_val_raw = inverse_target(y_val, features)

        if cfg_idx == 0:
            results.extend(evaluate_baselines(raw_ot, boundaries.val_starts, lookback, true_val_raw))

        model, history, best_metric = train_model(
            cfg, x_train, y_train, x_val, y_val, features, seed + cfg_idx + 1
        )
        trained_models[cfg["name"]] = model
        histories[cfg["name"]] = history
        results.append(
            {
                "name": cfg["name"],
                "kind": "mlp",
                "L": lookback,
                "hidden": int(cfg["hidden"]),
                "epochs": int(history[-1]["epoch"]),
                "mse": best_metric["mse"],
                "rmse": best_metric["rmse"],
                "mae": best_metric["mae"],
                "horizon_rmse": best_metric["horizon_rmse"],
            }
        )

    report_checks(
        "Model training",
        [
            ("train/validation losses were recorded", all(len(h) > 0 for h in histories.values())),
            ("at least one MLP model was trained", len(trained_models) > 0),
            ("horizon-wise RMSE is available", all(len(r["horizon_rmse"]) == HORIZON for r in results)),
            ("validation MSE is finite for all models", all(np.isfinite(float(r["mse"])) for r in results)),
        ],
    )
    return results, trained_models, histories


def write_validation_summary(path: Path, results: list[dict[str, Any]]) -> None:
    rows = sorted(results, key=lambda r: float(r["mse"]))
    fieldnames = ["rank", "name", "kind", "L", "hidden", "epochs", "mse", "rmse", "mae"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "name": row["name"],
                    "kind": row["kind"],
                    "L": row["L"],
                    "hidden": row["hidden"],
                    "epochs": row["epochs"],
                    "mse": f"{float(row['mse']):.8f}",
                    "rmse": f"{float(row['rmse']):.8f}",
                    "mae": f"{float(row['mae']):.8f}",
                }
            )


def write_histories(path: Path, histories: dict[str, list[dict[str, float]]]) -> None:
    fieldnames = ["model", "epoch", "train_loss_scaled", "val_mse_raw", "val_rmse_raw", "val_mae_raw"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, rows in histories.items():
            for row in rows:
                writer.writerow(
                    {
                        "model": model_name,
                        "epoch": int(row["epoch"]),
                        "train_loss_scaled": f"{row['train_loss_scaled']:.8f}",
                        "val_mse_raw": f"{row['val_mse_raw']:.8f}",
                        "val_rmse_raw": f"{row['val_rmse_raw']:.8f}",
                        "val_mae_raw": f"{row['val_mae_raw']:.8f}",
                    }
                )


def date_indices(dates: np.ndarray, sample_datetimes: np.ndarray) -> np.ndarray:
    index = {dt: i for i, dt in enumerate(dates)}
    return np.asarray([index[dt] for dt in sample_datetimes], dtype=np.int64)


def predict_submission(
    selected: dict[str, Any],
    model: MLPRegressor | None,
    bundle: DataBundle,
    features: FeatureBundle,
) -> np.ndarray:
    id_idx = date_indices(bundle.dates, bundle.sample_datetimes)
    raw_ot = features.features_raw[:, features.feature_names.index(TARGET_COL)]
    kind = selected["kind"]
    lookback = int(selected["L"])

    if kind == "baseline":
        if selected["name"] == "last_value":
            pred = np.repeat(raw_ot[id_idx - 1, None], HORIZON, axis=1)
        elif selected["name"] == "last_96h_repeat":
            idx = id_idx[:, None] - HORIZON + np.arange(HORIZON, dtype=np.int64)[None, :]
            pred = raw_ot[idx]
        else:
            raise ValueError(f"Unknown baseline for submission: {selected['name']}")
        return pred.astype(np.float32)

    if model is None:
        raise ValueError("MLP submission requires a trained model.")
    starts = id_idx - lookback
    if np.any(starts < 0):
        raise ValueError("Not enough history for at least one submission ID.")
    x_sub, _ = make_windows(
        features.features_scaled,
        features.features_scaled[:, features.feature_names.index(TARGET_COL)],
        starts,
        lookback,
        HORIZON,
    )
    pred_scaled = model.predict(x_sub)
    return inverse_target(pred_scaled, features).astype(np.float32)


def save_submission(
    path: Path,
    bundle: DataBundle,
    predictions: np.ndarray,
) -> None:
    if predictions.shape != (len(bundle.sample_ids), HORIZON):
        raise ValueError(f"Bad prediction shape: {predictions.shape}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(bundle.sample_columns)
        for sample_id, pred in zip(bundle.sample_ids, predictions):
            writer.writerow([sample_id, *[f"{float(x):.6f}" for x in pred]])


def validate_submission(
    bundle: DataBundle,
    predictions: np.ndarray,
    selected: dict[str, Any],
) -> None:
    id_idx = date_indices(bundle.dates, bundle.sample_datetimes)
    if selected["kind"] == "baseline":
        input_end_ok = bool(np.all(id_idx > 0))
    else:
        input_end_ok = bool(np.all(id_idx - int(selected["L"]) >= 0))
    print("\nSubmission preview:")
    for i in list(range(min(3, len(bundle.sample_ids)))) + list(range(max(3, len(bundle.sample_ids) - 3), len(bundle.sample_ids))):
        print(
            f"{bundle.sample_ids[i]}: "
            f"T0={predictions[i, 0]:.4f}, T1={predictions[i, 1]:.4f}, T95={predictions[i, -1]:.4f}"
        )
    print(
        "Submission value summary: "
        f"min={float(np.min(predictions)):.4f}, "
        f"max={float(np.max(predictions)):.4f}, "
        f"mean={float(np.mean(predictions)):.4f}, "
        f"std={float(np.std(predictions)):.4f}"
    )
    report_checks(
        "Submission generation",
        [
            ("prediction shape matches sample_submit.csv rows x 96", predictions.shape == (len(bundle.sample_ids), HORIZON)),
            ("ID order is preserved by construction", True),
            ("no NaN or inf values are present", bool(np.all(np.isfinite(predictions)))),
            ("each prediction uses only rows before its own ID timestamp", input_end_ok),
        ],
    )


def save_model_artifact(
    path: Path,
    model: MLPRegressor,
    selected: dict[str, Any],
    features: FeatureBundle,
    train_boundary_idx: int,
    test_start_idx: int,
) -> None:
    state = model.state_dict()
    metadata = {
        "selected": {k: v for k, v in selected.items() if k != "horizon_rmse"},
        "feature_names": features.feature_names,
        "target_col": TARGET_COL,
        "horizon": HORIZON,
        "train_boundary_idx": train_boundary_idx,
        "test_start_idx": test_start_idx,
        "test_start": TEST_START.strftime("%Y-%m-%d %H:%M:%S"),
    }
    np.savez(
        path,
        **state,
        feature_mean=features.mean,
        feature_std=features.std,
        target_mean=np.asarray([features.target_mean], dtype=np.float32),
        target_std=np.asarray([features.target_std], dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )


def write_leakage_note(
    path: Path,
    selected: dict[str, Any],
    train_boundary_idx: int,
    test_start_idx: int,
) -> None:
    text = f"""# Leakage note

- Training and validation target windows are restricted to rows before 2018-02-01 00:00:00.
- Scaler statistics are fit only on rows `[0, {train_boundary_idx})`, the training period.
- Validation target rows start at `{train_boundary_idx}` and end before `{test_start_idx}`.
- Rows from 2018-02-01 onward are not used for training, validation, or model selection.
- Submission inference uses only history before each sample ID timestamp. For later IDs, this can include earlier test-period observations because the project statement allows data before the target date as input.
- Final `submit.csv` method: `{selected['name']}` (`{selected['kind']}`), validation MSE `{float(selected['mse']):.6f}`.
"""
    path.write_text(text, encoding="utf-8")


def optimize_and_submit(
    bundle: DataBundle,
    features: FeatureBundle,
    results: list[dict[str, Any]],
    trained_models: dict[str, MLPRegressor],
    histories: dict[str, list[dict[str, float]]],
    train_boundary_idx: int,
    test_start_idx: int,
    output_dir: Path,
) -> None:
    stage("7. Model optimization and submission")
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=lambda r: float(r["mse"]))
    best_overall = sorted_results[0]
    mlp_results = [r for r in sorted_results if r["kind"] == "mlp"]
    best_mlp = mlp_results[0]

    print("Validation ranking:")
    for rank, row in enumerate(sorted_results, start=1):
        print(
            f"{rank}. {row['name']} ({row['kind']}): "
            f"L={row['L']} mse={float(row['mse']):.5f} rmse={float(row['rmse']):.5f}"
        )

    write_validation_summary(output_dir / "validation_summary.csv", results)
    write_histories(output_dir / "training_history.csv", histories)

    selected_model = trained_models.get(best_overall["name"])
    predictions = predict_submission(best_overall, selected_model, bundle, features)
    save_submission(output_dir / "submit.csv", bundle, predictions)
    validate_submission(bundle, predictions, best_overall)
    write_leakage_note(output_dir / "leakage_note.md", best_overall, train_boundary_idx, test_start_idx)

    if best_mlp["name"] in trained_models:
        save_model_artifact(
            output_dir / "model_best_mlp.npz",
            trained_models[best_mlp["name"]],
            best_mlp,
            features,
            train_boundary_idx,
            test_start_idx,
        )
        if best_mlp["name"] != best_overall["name"]:
            mlp_predictions = predict_submission(best_mlp, trained_models[best_mlp["name"]], bundle, features)
            save_submission(output_dir / "submit_mlp_best.csv", bundle, mlp_predictions)

    report_checks(
        "Model optimization",
        [
            ("validation summary was written", (output_dir / "validation_summary.csv").exists()),
            ("training history was written", (output_dir / "training_history.csv").exists()),
            ("submit.csv was written", (output_dir / "submit.csv").exists()),
            ("best MLP artifact was written", (output_dir / "model_best_mlp.npz").exists()),
            ("leakage note was written", (output_dir / "leakage_note.md").exists()),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ETTh1 96-hour OT forecaster and create submit.csv")
    parser.add_argument("--data-path", type=Path, default=Path("ETTh1.csv"))
    parser.add_argument("--sample-path", type=Path, default=Path("sample_submit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction-before-test", type=float, default=0.75)
    parser.add_argument("--full", action="store_true", help="Run a larger experiment grid.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for every MLP config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    set_seed(args.seed)

    bundle = load_data(args.data_path, args.sample_path)
    define_problem(bundle)
    acquire_and_validate_data(bundle)
    test_start_idx, train_boundary_idx = preprocess_data(bundle, args.train_fraction_before_test)
    configs = config_grid(full=args.full, epochs_override=args.epochs)
    features = feature_engineering(bundle, test_start_idx, train_boundary_idx, [int(c["L"]) for c in configs])
    model_selection(configs)
    results, trained_models, histories = run_training_and_optimization(
        bundle, features, test_start_idx, train_boundary_idx, configs, args.seed
    )
    optimize_and_submit(
        bundle,
        features,
        results,
        trained_models,
        histories,
        train_boundary_idx,
        test_start_idx,
        args.output_dir,
    )
    print(f"\nDone in {(time.time() - start_time):.1f} seconds.")


if __name__ == "__main__":
    main()
