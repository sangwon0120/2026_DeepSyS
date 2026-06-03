from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
DATA_PATH = "ETTh1.csv"
SAMPLE_PATH = "csvFiles/sample_submit.csv"
OUTPUT_PATH = "csvFiles/submit.csv"
BEST_MODEL_PATH = "best_model.pt"
RESIDUAL_MODEL_PATH = "best_seq_residual_booster.pt"
PROJECT_DIR = Path("/content/drive/MyDrive/2026_DeepSyS")

TEST_START = pd.Timestamp("2018-02-01 00:00:00")
TARGET_COL = "OT"
RAW_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
FEATURE_COLS = RAW_COLS + TIME_COLS
OT_IDX = FEATURE_COLS.index(TARGET_COL)
RAW_IDXS = [FEATURE_COLS.index(c) for c in RAW_COLS]
TIME_IDXS = [FEATURE_COLS.index(c) for c in TIME_COLS]

MSE_TARGET = 10.0
SHORT_LOOKBACK = 168
SEQ_LOOKBACK = 672
LOOKBACK = SHORT_LOOKBACK
HORIZON = 96
TRAIN_FRACTION_BEFORE_TEST = 0.75

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 128
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 6
GRAD_CLIP = 1.0
LOG_INTERVAL = 20
SEQ_HIDDEN_SIZE = 128
SEQ_NUM_LAYERS = 2
SEQ_DROPOUT = 0.2
AUX_HIDDEN_SIZE = 128
SEQ_ERROR_LAGS = (96, 168, 336)
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 60
RESIDUAL_BACKTEST_EPOCHS = 40
RESIDUAL_LAMBDAS = tuple(float(x) for x in np.linspace(0.0, 1.0, 21))

PATCH_MODEL_PATH = "best_patchtst_residual.pt"
PATCH_LOOKBACK = 672
PATCH_EXTRA_LAG = 168
PATCH_START_LOOKBACK = PATCH_LOOKBACK + PATCH_EXTRA_LAG
PATCH_LEN = 24
PATCH_STRIDE = 12
PATCH_D_MODEL = 128
PATCH_N_HEADS = 8
PATCH_ENCODER_LAYERS = 3
PATCH_D_FF = 256
PATCH_DROPOUT = 0.2
PATCH_AUX_HIDDEN_SIZE = 128
PATCH_BATCH_SIZE = 64
PATCH_PRETRAIN_EPOCHS = 20
PATCH_FINETUNE_EPOCHS = 60
PATCH_BACKTEST_EPOCHS = 40
PATCH_PATIENCE = 6
PATCH_FEATURE_CLIP_IQR = 3.0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_file(path: str | Path, filename: str) -> Path:
    path = Path(path)
    if path.exists():
        return path

    drive_path = PROJECT_DIR / filename
    if drive_path.exists():
        return drive_path

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.rglob(filename))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"{filename} not found. Put it in the notebook directory, {PROJECT_DIR}, or /kaggle/input.")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["date"]
    out["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)
    out["doy_sin"] = np.sin(2 * np.pi * (dt.dt.dayofyear - 1) / 366)
    out["doy_cos"] = np.cos(2 * np.pi * (dt.dt.dayofyear - 1) / 366)
    return out


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    data_path = find_file(DATA_PATH, "ETTh1.csv")
    sample_path = find_file(SAMPLE_PATH, "sample_submit.csv")

    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    sub = pd.read_csv(sample_path)
    id_col = "ID" if "ID" in sub.columns else "timestamp"
    sub[id_col] = pd.to_datetime(sub[id_col])

    print("data:", data_path, df.shape, df["date"].min(), "~", df["date"].max())
    print("sample:", sample_path, sub.shape, sub[id_col].min(), "~", sub[id_col].max())

    assert set(["date", *RAW_COLS]).issubset(df.columns)
    assert [f"T{i}" for i in range(HORIZON)] == [c for c in sub.columns if c.startswith("T")]
    assert df["date"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert df.isna().sum().sum() == 0
    return df, sub, id_col


def load_patchtst_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    data_path = find_file(DATA_PATH, "ETTh1.csv")
    sample_path = find_file(SAMPLE_PATH, "sample_submit.csv")

    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    sub = pd.read_csv(sample_path)
    id_col = "ID" if "ID" in sub.columns else "timestamp"
    sub[id_col] = pd.to_datetime(sub[id_col])

    print("patch data:", data_path, df.shape, df["date"].min(), "~", df["date"].max())
    print("sample:", sample_path, sub.shape, sub[id_col].min(), "~", sub[id_col].max())

    assert set(["date", *RAW_COLS]).issubset(df.columns)
    assert [f"T{i}" for i in range(HORIZON)] == [c for c in sub.columns if c.startswith("T")]
    return df, sub, id_col


def midnight_target_indices(df: pd.DataFrame, start_idx: int, end_idx: int) -> np.ndarray:
    targets = np.arange(start_idx, end_idx - HORIZON + 1)
    targets = targets[df.loc[targets, "date"].dt.hour.values == 0]
    return targets.astype(np.int64)


def all_hour_target_indices(start_idx: int, end_idx: int) -> np.ndarray:
    return np.arange(start_idx, end_idx - HORIZON + 1, dtype=np.int64)


def clean_hourly_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", *RAW_COLS]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    full_dates = pd.date_range(out["date"].min(), out["date"].max(), freq="h")
    out = out.set_index("date").reindex(full_dates)
    out.index.name = "date"
    out[RAW_COLS] = out[RAW_COLS].astype("float64").interpolate(method="linear", limit_direction="both").ffill().bfill()
    out = out.reset_index()

    assert out["date"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert out[RAW_COLS].isna().sum().sum() == 0
    return out


def fit_feature_clip_bounds(df: pd.DataFrame, train_end_idx: int) -> dict[str, tuple[float, float]]:
    train = df.loc[: train_end_idx - 1, RAW_COLS]
    bounds = {}
    for col in RAW_COLS:
        q1 = float(train[col].quantile(0.25))
        q3 = float(train[col].quantile(0.75))
        iqr = q3 - q1
        if iqr == 0.0 or not np.isfinite(iqr):
            lower = float(train[col].min())
            upper = float(train[col].max())
        else:
            lower = q1 - PATCH_FEATURE_CLIP_IQR * iqr
            upper = q3 + PATCH_FEATURE_CLIP_IQR * iqr
        bounds[col] = (lower, upper)
    return bounds


def apply_feature_clip(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lower, upper) in bounds.items():
        out[col] = out[col].clip(lower, upper)
    return out


def prepare_data(df: pd.DataFrame) -> dict:
    df = add_time_features(df)

    test_start_idx = int(np.where(df["date"].values == np.datetime64(TEST_START))[0][0])
    train_end_idx = int(test_start_idx * TRAIN_FRACTION_BEFORE_TEST)
    train_end_idx = (train_end_idx // 24) * 24

    train_targets = midnight_target_indices(df, LOOKBACK, train_end_idx)
    seq_train_targets = midnight_target_indices(df, SEQ_LOOKBACK, train_end_idx)
    pretrain_targets = all_hour_target_indices(SEQ_LOOKBACK, train_end_idx)
    val_targets = midnight_target_indices(df, train_end_idx, test_start_idx)

    print("train target:", df.loc[train_targets[0], "date"], "~", df.loc[train_targets[-1] + HORIZON - 1, "date"])
    print("seq train target:", df.loc[seq_train_targets[0], "date"], "~", df.loc[seq_train_targets[-1] + HORIZON - 1, "date"])
    print("pretrain target:", df.loc[pretrain_targets[0], "date"], "~", df.loc[pretrain_targets[-1] + HORIZON - 1, "date"])
    print("val target:", df.loc[val_targets[0], "date"], "~", df.loc[val_targets[-1] + HORIZON - 1, "date"])
    print("test starts:", df.loc[test_start_idx, "date"])

    mean = df.loc[: train_end_idx - 1, FEATURE_COLS].mean()
    std = df.loc[: train_end_idx - 1, FEATURE_COLS].std().replace(0, 1.0)
    values_scaled = ((df[FEATURE_COLS] - mean) / std).astype("float32").values
    target_scaled = values_scaled[:, OT_IDX]

    print("feature shape:", values_scaled.shape)
    print(
        "pretrain targets:",
        len(pretrain_targets),
        "train targets:",
        len(train_targets),
        "seq train targets:",
        len(seq_train_targets),
        "val targets:",
        len(val_targets),
    )

    assert len(pretrain_targets) > 0 and len(train_targets) > 0 and len(seq_train_targets) > 0 and len(val_targets) > 0
    assert (df.loc[train_targets, "date"].dt.hour == 0).all()
    assert (df.loc[seq_train_targets, "date"].dt.hour == 0).all()
    assert (df.loc[val_targets, "date"].dt.hour == 0).all()
    assert train_targets[0] - LOOKBACK >= 0
    assert pretrain_targets[0] - SEQ_LOOKBACK >= 0
    assert seq_train_targets[0] - SEQ_LOOKBACK >= 0
    assert val_targets[0] - SEQ_LOOKBACK >= 0
    assert pretrain_targets[-1] + HORIZON <= train_end_idx
    assert train_targets[-1] + HORIZON <= train_end_idx
    assert seq_train_targets[-1] + HORIZON <= train_end_idx
    assert val_targets[0] >= train_end_idx
    assert val_targets[-1] + HORIZON <= test_start_idx
    assert (df.loc[val_targets - 1, "date"].values < df.loc[val_targets, "date"].values).all()

    return {
        "df": df,
        "mean": mean,
        "std": std,
        "values_scaled": values_scaled,
        "target_scaled": target_scaled,
        "pretrain_targets": pretrain_targets,
        "train_targets": train_targets,
        "seq_train_targets": seq_train_targets,
        "val_targets": val_targets,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
    }


def make_data_view(data: dict, train_end_idx: int, train_targets: np.ndarray, val_targets: np.ndarray) -> dict:
    df = data["df"]
    mean = df.loc[: train_end_idx - 1, FEATURE_COLS].mean()
    std = df.loc[: train_end_idx - 1, FEATURE_COLS].std().replace(0, 1.0)
    values_scaled = ((df[FEATURE_COLS] - mean) / std).astype("float32").values
    pretrain_targets = all_hour_target_indices(SEQ_LOOKBACK, train_end_idx)
    seq_train_targets = midnight_target_indices(df, SEQ_LOOKBACK, train_end_idx)

    return {
        "df": df,
        "mean": mean,
        "std": std,
        "values_scaled": values_scaled,
        "target_scaled": values_scaled[:, OT_IDX],
        "pretrain_targets": pretrain_targets,
        "train_targets": train_targets,
        "seq_train_targets": seq_train_targets,
        "val_targets": val_targets,
        "train_end_idx": train_end_idx,
        "test_start_idx": data["test_start_idx"],
    }


def prepare_patchtst_data(
    df: pd.DataFrame,
    train_end_idx: int | None = None,
    train_targets: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
) -> dict:
    raw_df = clean_hourly_numeric_frame(df)
    test_start_idx = int(np.where(raw_df["date"].values == np.datetime64(TEST_START))[0][0])
    if train_end_idx is None:
        train_end_idx = int(test_start_idx * TRAIN_FRACTION_BEFORE_TEST)
        train_end_idx = (train_end_idx // 24) * 24

    if train_targets is None:
        train_targets = midnight_target_indices(raw_df, LOOKBACK, train_end_idx)
    patch_train_targets = midnight_target_indices(raw_df, PATCH_START_LOOKBACK, train_end_idx)
    patch_pretrain_targets = all_hour_target_indices(PATCH_START_LOOKBACK, train_end_idx)
    if val_targets is None:
        val_targets = midnight_target_indices(raw_df, train_end_idx, test_start_idx)

    clip_bounds = fit_feature_clip_bounds(raw_df, train_end_idx)
    clipped_df = apply_feature_clip(raw_df, clip_bounds)
    df_with_time = add_time_features(raw_df)
    feature_df = add_time_features(clipped_df)

    mean = df_with_time.loc[: train_end_idx - 1, FEATURE_COLS].mean()
    std = df_with_time.loc[: train_end_idx - 1, FEATURE_COLS].std().replace(0, 1.0)
    feature_mean = feature_df.loc[: train_end_idx - 1, FEATURE_COLS].mean()
    feature_std = feature_df.loc[: train_end_idx - 1, FEATURE_COLS].std().replace(0, 1.0)

    values_scaled = ((df_with_time[FEATURE_COLS] - mean) / std).astype("float32").values
    patch_values_scaled = ((feature_df[FEATURE_COLS] - feature_mean) / feature_std).astype("float32").values

    print("patch feature shape:", patch_values_scaled.shape)
    print(
        "patch pretrain targets:",
        len(patch_pretrain_targets),
        "baseline train targets:",
        len(train_targets),
        "patch train targets:",
        len(patch_train_targets),
        "val targets:",
        len(val_targets),
    )

    assert len(patch_pretrain_targets) > 0 and len(train_targets) > 0 and len(patch_train_targets) > 0 and len(val_targets) > 0
    assert (df_with_time.loc[train_targets, "date"].dt.hour == 0).all()
    assert (df_with_time.loc[patch_train_targets, "date"].dt.hour == 0).all()
    assert (df_with_time.loc[val_targets, "date"].dt.hour == 0).all()
    assert train_targets[0] - LOOKBACK >= 0
    assert patch_pretrain_targets[0] - PATCH_START_LOOKBACK >= 0
    assert patch_train_targets[0] - PATCH_START_LOOKBACK >= 0
    assert val_targets[0] - PATCH_START_LOOKBACK >= 0
    assert patch_pretrain_targets[-1] + HORIZON <= train_end_idx
    assert train_targets[-1] + HORIZON <= train_end_idx
    assert patch_train_targets[-1] + HORIZON <= train_end_idx
    assert val_targets[-1] + HORIZON <= test_start_idx

    return {
        "raw_df": raw_df,
        "df": df_with_time,
        "mean": mean,
        "std": std,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "clip_bounds": clip_bounds,
        "values_scaled": values_scaled,
        "patch_values_scaled": patch_values_scaled,
        "target_scaled": values_scaled[:, OT_IDX],
        "train_targets": train_targets,
        "patch_train_targets": patch_train_targets,
        "patch_pretrain_targets": patch_pretrain_targets,
        "val_targets": val_targets,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
    }


def make_patchtst_data_view(data: dict, train_end_idx: int, train_targets: np.ndarray, val_targets: np.ndarray) -> dict:
    return prepare_patchtst_data(data["raw_df"], train_end_idx=train_end_idx, train_targets=train_targets, val_targets=val_targets)


def inverse_ot(x: np.ndarray, mean: pd.Series, std: pd.Series) -> np.ndarray:
    return x * float(std[TARGET_COL]) + float(mean[TARGET_COL])


def score_raw(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    mse = float(np.mean(err**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "horizon_rmse": np.sqrt(np.mean(err**2, axis=0)),
    }


def make_true(raw_ot: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    return np.stack([raw_ot[t : t + HORIZON] for t in target_indices])


def predict_last_value(raw_ot: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    return np.repeat(raw_ot[target_indices - 1, None], HORIZON, axis=1)


def predict_last_96h_repeat(raw_ot: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    assert np.all(target_indices - HORIZON >= 0)
    return np.stack([raw_ot[t - HORIZON : t] for t in target_indices])


def predict_last_week_same_time(raw_ot: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    assert np.all(target_indices - 168 >= 0)
    return np.stack([raw_ot[t - 168 : t - 168 + HORIZON] for t in target_indices])


def baseline_prediction(raw_ot: np.ndarray, target_indices: np.ndarray, predictor: dict) -> np.ndarray:
    name = predictor["name"]
    if name == "last_value":
        return predict_last_value(raw_ot, target_indices)
    if name == "last_96h_repeat":
        return predict_last_96h_repeat(raw_ot, target_indices)
    if name == "last_week_same_time":
        return predict_last_week_same_time(raw_ot, target_indices)
    if name.startswith("blend_"):
        alpha = predictor["alpha"]
        pred_a = baseline_prediction(raw_ot, target_indices, {"name": predictor["a"]})
        pred_b = baseline_prediction(raw_ot, target_indices, {"name": predictor["b"]})
        if np.isscalar(alpha):
            return alpha * pred_a + (1.0 - alpha) * pred_b
        alpha = np.asarray(alpha, dtype=np.float32)[None, :]
        return alpha * pred_a + (1.0 - alpha) * pred_b
    if name.startswith("threeway_"):
        weights = np.asarray(predictor["weights"], dtype=np.float32)
        pred_a = predict_last_value(raw_ot, target_indices)
        pred_b = predict_last_96h_repeat(raw_ot, target_indices)
        pred_c = predict_last_week_same_time(raw_ot, target_indices)
        return weights[0][None, :] * pred_a + weights[1][None, :] * pred_b + weights[2][None, :] * pred_c
    if name.startswith("ensemble_"):
        weight = float(predictor["weight"])
        pred_a = baseline_prediction(raw_ot, target_indices, predictor["a_predictor"])
        pred_b = baseline_prediction(raw_ot, target_indices, predictor["b_predictor"])
        return weight * pred_a + (1.0 - weight) * pred_b
    raise ValueError(f"unknown baseline predictor: {name}")


def fit_blend(train_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    diff = pred_a - pred_b
    denom = float(np.sum(diff * diff))
    if denom == 0.0:
        return 1.0
    alpha = float(np.sum((train_true - pred_b) * diff) / denom)
    return float(np.clip(alpha, 0.0, 1.0))


def fit_block_blend(train_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    alpha = np.zeros(HORIZON, dtype=np.float32)
    for start in range(0, HORIZON, 24):
        end = start + 24
        block_alpha = fit_blend(train_true[:, start:end], pred_a[:, start:end], pred_b[:, start:end])
        alpha[start:end] = block_alpha
    return alpha


def alpha_summary(alpha) -> str:
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        return ""
    if np.isscalar(alpha):
        return f"{float(alpha):.6f}"
    alpha = np.asarray(alpha)
    return "[" + ", ".join(f"{float(x):.4f}" for x in alpha[::24]) + "]"


def fit_threeway_block(
    train_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    pred_c: np.ndarray,
    block_size: int = 24,
) -> np.ndarray:
    weights = np.zeros((3, HORIZON), dtype=np.float32)
    grid = np.linspace(0.0, 1.0, 21)
    for start in range(0, HORIZON, block_size):
        end = min(start + block_size, HORIZON)
        best_loss = float("inf")
        best_weights = (1.0, 0.0, 0.0)
        y = train_true[:, start:end]
        a = pred_a[:, start:end]
        b = pred_b[:, start:end]
        c = pred_c[:, start:end]
        for w_a in grid:
            for w_b in grid:
                w_c = 1.0 - w_a - w_b
                if w_c < 0.0:
                    continue
                pred = w_a * a + w_b * b + w_c * c
                loss = float(np.mean((pred - y) ** 2))
                if loss < best_loss:
                    best_loss = loss
                    best_weights = (float(w_a), float(w_b), float(w_c))
        weights[:, start:end] = np.asarray(best_weights, dtype=np.float32)[:, None]
    return weights


def shrink_weights(fine_weights: np.ndarray, base_weights: np.ndarray, fine_amount: float) -> np.ndarray:
    return (fine_amount * fine_weights + (1.0 - fine_amount) * base_weights).astype(np.float32)


def weights_summary(weights) -> str:
    if weights is None:
        return ""
    weights = np.asarray(weights)
    starts = [0]
    for i in range(1, HORIZON):
        if not np.allclose(weights[:, i], weights[:, i - 1]):
            starts.append(i)
    return "[" + ", ".join(f"T{i}:({weights[0, i]:.2f},{weights[1, i]:.2f},{weights[2, i]:.2f})" for i in starts) + "]"


def build_baseline_candidates(raw_ot: np.ndarray, train_targets: np.ndarray) -> list[dict]:
    train_true = make_true(raw_ot, train_targets)

    candidates = [
        {"name": "last_value", "kind": "baseline"},
        {"name": "last_96h_repeat", "kind": "baseline"},
        {"name": "last_week_same_time", "kind": "baseline"},
    ]

    blend_pairs = [
        ("blend_last_value_last_week", "last_value", "last_week_same_time"),
        ("blend_last_value_last_96h", "last_value", "last_96h_repeat"),
    ]
    for name, a_name, b_name in blend_pairs:
        pred_a = baseline_prediction(raw_ot, train_targets, {"name": a_name})
        pred_b = baseline_prediction(raw_ot, train_targets, {"name": b_name})
        global_alpha = fit_blend(train_true, pred_a, pred_b)
        block_alpha = fit_block_blend(train_true, pred_a, pred_b)
        candidates.append({"name": name, "kind": "baseline", "a": a_name, "b": b_name, "alpha": global_alpha})
        candidates.append(
            {
                "name": f"{name}_4block",
                "kind": "baseline",
                "a": a_name,
                "b": b_name,
                "alpha": block_alpha,
            }
        )
        for shrink in (0.25, 0.50, 0.75):
            alpha = global_alpha + shrink * (block_alpha - global_alpha)
            candidates.append(
                {
                    "name": f"{name}_4block_shrink{int(shrink * 100)}",
                    "kind": "baseline",
                    "a": a_name,
                    "b": b_name,
                    "alpha": alpha.astype(np.float32),
                }
            )

    pred_last = predict_last_value(raw_ot, train_targets)
    pred_prev = predict_last_96h_repeat(raw_ot, train_targets)
    pred_week = predict_last_week_same_time(raw_ot, train_targets)
    weights_4block = fit_threeway_block(train_true, pred_last, pred_prev, pred_week, block_size=24)
    threeway_predictor = {
        "name": "threeway_last_prev_week_4block",
        "kind": "baseline",
        "weights": weights_4block,
    }
    candidates.append(threeway_predictor)

    for name, block_size in [("8block", 12), ("12block", 8), ("24block", 4)]:
        fine_weights = fit_threeway_block(train_true, pred_last, pred_prev, pred_week, block_size=block_size)
        candidates.append(
            {
                "name": f"threeway_last_prev_week_{name}",
                "kind": "baseline",
                "weights": fine_weights,
            }
        )
        for shrink in (0.50, 0.75):
            candidates.append(
                {
                    "name": f"threeway_last_prev_week_{name}_shrink{int(shrink * 100)}",
                    "kind": "baseline",
                    "weights": shrink_weights(fine_weights, weights_4block, shrink),
                }
            )

    global_week_predictor = next(candidate for candidate in candidates if candidate["name"] == "blend_last_value_last_week")
    for weight in (0.25, 0.50, 0.75):
        candidates.append(
            {
                "name": f"ensemble_threeway_globalblend_w{int(weight * 100)}",
                "kind": "baseline",
                "weight": weight,
                "a_predictor": dict(threeway_predictor),
                "b_predictor": dict(global_week_predictor),
            }
        )
    return candidates


def evaluate_baseline_candidates(raw_ot: np.ndarray, candidates: list[dict], val_targets: np.ndarray) -> tuple[pd.DataFrame, dict]:
    val_true = make_true(raw_ot, val_targets)

    rows = []
    best_predictor = None
    best_mse = float("inf")
    for predictor in candidates:
        pred = baseline_prediction(raw_ot, val_targets, predictor)
        metric = score_raw(pred, val_true)
        row = {
            "name": predictor["name"],
            "mse": metric["mse"],
            "rmse": metric["rmse"],
            "mae": metric["mae"],
            "alpha": alpha_summary(predictor.get("alpha", np.nan)),
            "weights": weights_summary(predictor.get("weights")),
            "weight": predictor.get("weight", ""),
        }
        rows.append(row)
        predictor["metric"] = metric
        if metric["mse"] < best_mse:
            best_mse = metric["mse"]
            best_predictor = dict(predictor)

    baseline_df = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    assert best_predictor is not None
    assert best_mse <= float(baseline_df.loc[baseline_df["name"] == "last_value", "mse"].iloc[0])
    return baseline_df, best_predictor


def fit_baseline_candidates(data: dict) -> tuple[pd.DataFrame, dict]:
    raw_ot = data["df"][TARGET_COL].values.astype("float32")
    candidates = build_baseline_candidates(raw_ot, data["train_targets"])
    return evaluate_baseline_candidates(raw_ot, candidates, data["val_targets"])


def evaluate_baseline_on_targets(data: dict, predictor: dict, target_indices: np.ndarray) -> dict:
    raw_ot = data["df"][TARGET_COL].values.astype("float32")
    return score_raw(baseline_prediction(raw_ot, target_indices, predictor), make_true(raw_ot, target_indices))


def rolling_backtest(data: dict) -> pd.DataFrame:
    df = data["df"]
    raw_ot = df[TARGET_COL].values.astype("float32")
    train_end_idx = data["train_end_idx"]
    fold_ends = [
        train_end_idx - 24 * 90,
        train_end_idx - 24 * 45,
        train_end_idx,
    ]
    rows = []
    for fold_idx, val_start in enumerate(fold_ends, start=1):
        val_end = val_start + 24 * 142
        if val_end > data["test_start_idx"]:
            continue
        train_targets = midnight_target_indices(df, LOOKBACK, val_start)
        val_targets = midnight_target_indices(df, val_start, val_end)
        if len(train_targets) == 0 or len(val_targets) == 0:
            continue
        candidates = build_baseline_candidates(raw_ot, train_targets)
        fold_df, _ = evaluate_baseline_candidates(raw_ot, candidates, val_targets)
        for row in fold_df.to_dict("records"):
            rows.append(
                {
                    "fold": fold_idx,
                    "val_start": df.loc[val_targets[0], "date"],
                    "val_end": df.loc[val_targets[-1] + HORIZON - 1, "date"],
                    "name": row["name"],
                    "mse": row["mse"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                }
            )
    if not rows:
        return pd.DataFrame()
    fold_scores = pd.DataFrame(rows)
    summary = (
        fold_scores.groupby("name", as_index=False)
        .agg(mean_mse=("mse", "mean"), std_mse=("mse", "std"), max_mse=("mse", "max"), mean_mae=("mae", "mean"), folds=("mse", "count"))
        .sort_values(["mean_mse", "max_mse"])
        .reset_index(drop=True)
    )
    print("rolling backtest summary")
    print(summary)
    return summary


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, target: np.ndarray, target_indices: np.ndarray):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.target_indices = torch.tensor(target_indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.target_indices[idx])
        x = self.values[t - LOOKBACK : t]
        y = self.target[t : t + HORIZON]
        return x, y


class SeqResidualDataset(Dataset):
    def __init__(self, seq_features: np.ndarray, aux_features: np.ndarray, residual: np.ndarray):
        self.seq_features = torch.from_numpy(seq_features.astype("float32", copy=False))
        self.aux_features = torch.from_numpy(aux_features.astype("float32", copy=False))
        self.residual = torch.from_numpy(residual.astype("float32", copy=False))

    def __len__(self) -> int:
        return len(self.seq_features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.seq_features[idx], self.aux_features[idx], self.residual[idx]


class GRUForecast(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.out = nn.Linear(HIDDEN_SIZE, HORIZON)
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            self.out,
        )
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        residual = self.head(out[:, -1])
        baseline = x[:, -1, OT_IDX].unsqueeze(1).expand(-1, HORIZON)
        return baseline + residual


class SeqResidualBooster(nn.Module):
    def __init__(self, seq_input_dim: int, aux_input_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_input_dim,
            hidden_size=SEQ_HIDDEN_SIZE,
            num_layers=SEQ_NUM_LAYERS,
            batch_first=True,
            dropout=SEQ_DROPOUT if SEQ_NUM_LAYERS > 1 else 0.0,
        )
        self.attn = nn.Linear(SEQ_HIDDEN_SIZE, 1)
        self.aux_net = nn.Sequential(
            nn.LayerNorm(aux_input_dim),
            nn.Linear(aux_input_dim, AUX_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(SEQ_DROPOUT),
        )
        head_input_dim = SEQ_HIDDEN_SIZE * 2 + AUX_HIDDEN_SIZE
        self.out = nn.Linear(SEQ_HIDDEN_SIZE * 2, HORIZON)
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, SEQ_HIDDEN_SIZE * 2),
            nn.ReLU(),
            nn.Dropout(SEQ_DROPOUT),
            self.out,
        )
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, seq_x: torch.Tensor, aux_x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.gru(seq_x)
        attn_weight = torch.softmax(self.attn(seq_out).squeeze(-1), dim=1)
        context = torch.sum(seq_out * attn_weight.unsqueeze(-1), dim=1)
        last_hidden = seq_out[:, -1]
        aux_hidden = self.aux_net(aux_x)
        return self.head(torch.cat([last_hidden, context, aux_hidden], dim=1))


class PatchResidualDataset(Dataset):
    def __init__(self, data: dict, baseline_predictor: dict, target_indices: np.ndarray, include_target: bool = True):
        self.data = data
        self.values = data["patch_values_scaled"]
        self.target_scaled = data["target_scaled"]
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.include_target = include_target

        raw_ot = data["df"][TARGET_COL].values.astype("float32")
        baseline_raw = baseline_prediction(raw_ot, self.target_indices, baseline_predictor).astype("float32")
        ot_mean = float(data["mean"][TARGET_COL])
        ot_std = float(data["std"][TARGET_COL])
        self.baseline_scaled = ((baseline_raw - ot_mean) / ot_std).astype("float32")

        if include_target:
            true_scaled = np.stack([self.target_scaled[t : t + HORIZON] for t in self.target_indices]).astype("float32")
            self.residual = (true_scaled - self.baseline_scaled).astype("float32")
        else:
            self.residual = np.zeros_like(self.baseline_scaled, dtype=np.float32)

        assert np.all(self.target_indices - PATCH_START_LOOKBACK >= 0)
        assert np.all(self.target_indices + HORIZON <= len(self.values))

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.target_indices[idx])
        values = self.values

        recent_seq = values[t - PATCH_LOOKBACK : t]
        daily_delta = values[t - PATCH_LOOKBACK : t, RAW_IDXS] - values[t - PATCH_LOOKBACK - 24 : t - 24, RAW_IDXS]
        weekly_delta = values[t - PATCH_LOOKBACK : t, RAW_IDXS] - values[t - PATCH_LOOKBACK - 168 : t - 168, RAW_IDXS]
        seq_x = np.concatenate([recent_seq, daily_delta, weekly_delta], axis=1).astype("float32")

        future_time = values[t : t + HORIZON, TIME_IDXS].reshape(-1)
        raw_stats = np.concatenate([_raw_window_stats(values[t - length : t, RAW_IDXS]) for length in (24, 168, 336, 672)])
        recent_ot = values[t - SHORT_LOOKBACK : t, OT_IDX]
        daily_residual = recent_ot - values[t - SHORT_LOOKBACK - 24 : t - 24, OT_IDX]
        weekly_residual = recent_ot - values[t - SHORT_LOOKBACK - 168 : t - 168, OT_IDX]
        seasonal_residual_stats = np.concatenate([_residual_stats(daily_residual), _residual_stats(weekly_residual)])
        aux_x = np.concatenate([self.baseline_scaled[idx], future_time, raw_stats, seasonal_residual_stats]).astype("float32")

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(aux_x),
            torch.from_numpy(self.residual[idx]),
        )


class PatchTSTResidual(nn.Module):
    def __init__(
        self,
        seq_input_dim: int,
        aux_input_dim: int,
        context_length: int = PATCH_LOOKBACK,
        patch_len: int = PATCH_LEN,
        stride: int = PATCH_STRIDE,
        d_model: int = PATCH_D_MODEL,
        n_heads: int = PATCH_N_HEADS,
        encoder_layers: int = PATCH_ENCODER_LAYERS,
        d_ff: int = PATCH_D_FF,
        dropout: float = PATCH_DROPOUT,
    ):
        super().__init__()
        assert (context_length - patch_len) % stride == 0
        self.seq_input_dim = seq_input_dim
        self.context_length = context_length
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = 1 + (context_length - patch_len) // stride

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.channel_attn = nn.Linear(d_model, 1)
        self.aux_net = nn.Sequential(
            nn.LayerNorm(aux_input_dim),
            nn.Linear(aux_input_dim, PATCH_AUX_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        head_input_dim = d_model * 2 + PATCH_AUX_HIDDEN_SIZE
        self.out = nn.Linear(d_model * 2, HORIZON)
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            self.out,
        )
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, seq_x: torch.Tensor, aux_x: torch.Tensor) -> torch.Tensor:
        batch_size, context_length, channels = seq_x.shape
        assert context_length == self.context_length
        assert channels == self.seq_input_dim

        patches = seq_x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(batch_size * channels, self.num_patches, self.patch_len)
        tokens = self.patch_proj(patches) + self.pos_embed
        encoded = self.encoder(tokens).mean(dim=1).view(batch_size, channels, -1)

        attn_weight = torch.softmax(self.channel_attn(encoded).squeeze(-1), dim=1)
        attn_context = torch.sum(encoded * attn_weight.unsqueeze(-1), dim=1)
        mean_context = encoded.mean(dim=1)
        aux_hidden = self.aux_net(aux_x)
        return self.head(torch.cat([mean_context, attn_context, aux_hidden], dim=1))


def _raw_window_stats(window: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
        ]
    )


def _residual_stats(residual: np.ndarray) -> np.ndarray:
    recent_24 = residual[-24:]
    return np.asarray(
        [
            residual.mean(),
            residual.std(),
            residual.min(),
            residual.max(),
            residual[-1],
            recent_24.mean(),
            recent_24.std(),
        ],
        dtype=np.float32,
    )


def build_past_baseline_error_features(
    raw_ot: np.ndarray,
    baseline_predictor: dict,
    target_indices: np.ndarray,
    ot_std: float,
) -> np.ndarray:
    target_indices = np.asarray(target_indices, dtype=np.int64)
    features = []
    for lag in SEQ_ERROR_LAGS:
        past_targets = target_indices - lag
        assert np.all(past_targets >= 0)
        assert np.all(past_targets + HORIZON <= target_indices)

        past_true = make_true(raw_ot, past_targets)
        past_baseline = baseline_prediction(raw_ot, past_targets, baseline_predictor)
        past_error_scaled = ((past_true - past_baseline) / ot_std).astype("float32")
        past_error_stats = np.stack([_residual_stats(row) for row in past_error_scaled]).astype("float32")
        features.extend([past_error_scaled, past_error_stats])
    return np.concatenate(features, axis=1).astype("float32")


def build_seq_residual_arrays(
    data: dict,
    baseline_predictor: dict,
    target_indices: np.ndarray,
    include_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    df = data["df"]
    values = data["values_scaled"]
    raw_ot = df[TARGET_COL].values.astype("float32")
    ot_mean = float(data["mean"][TARGET_COL])
    ot_std = float(data["std"][TARGET_COL])
    target_indices = np.asarray(target_indices, dtype=np.int64)

    baseline_raw = baseline_prediction(raw_ot, target_indices, baseline_predictor).astype("float32")
    baseline_scaled = ((baseline_raw - ot_mean) / ot_std).astype("float32")
    past_baseline_error_features = build_past_baseline_error_features(raw_ot, baseline_predictor, target_indices, ot_std)

    assert np.all(target_indices - SEQ_LOOKBACK >= 0)
    assert np.all(target_indices + HORIZON <= len(values))

    seq_features = []
    aux_features = []
    for row_idx, t in enumerate(target_indices):
        recent_seq = values[t - SHORT_LOOKBACK : t]
        daily_delta = values[t - SHORT_LOOKBACK : t, RAW_IDXS] - values[t - SHORT_LOOKBACK - 24 : t - 24, RAW_IDXS]
        weekly_delta = values[t - SHORT_LOOKBACK : t, RAW_IDXS] - values[t - SHORT_LOOKBACK - 168 : t - 168, RAW_IDXS]
        seq_features.append(np.concatenate([recent_seq, daily_delta, weekly_delta], axis=1))

        future_time = values[t : t + HORIZON, TIME_IDXS].reshape(-1)
        raw_stats = np.concatenate([_raw_window_stats(values[t - length : t, RAW_IDXS]) for length in (24, 168, 336, 672)])

        recent_ot = values[t - SHORT_LOOKBACK : t, OT_IDX]
        daily_residual = recent_ot - values[t - SHORT_LOOKBACK - 24 : t - 24, OT_IDX]
        weekly_residual = recent_ot - values[t - SHORT_LOOKBACK - 168 : t - 168, OT_IDX]
        seasonal_residual_stats = np.concatenate([_residual_stats(daily_residual), _residual_stats(weekly_residual)])

        aux_features.append(
            np.concatenate(
                [
                    baseline_scaled[row_idx],
                    future_time,
                    raw_stats,
                    seasonal_residual_stats,
                    past_baseline_error_features[row_idx],
                ]
            )
        )

    seq_x = np.asarray(seq_features, dtype=np.float32)
    aux_x = np.asarray(aux_features, dtype=np.float32)
    if not include_target:
        return seq_x, aux_x, None, baseline_scaled

    true_scaled = np.stack([data["target_scaled"][t : t + HORIZON] for t in target_indices]).astype("float32")
    residual = (true_scaled - baseline_scaled).astype("float32")
    return seq_x, aux_x, residual, baseline_scaled


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def metric(pred_scaled: np.ndarray, true_scaled: np.ndarray, mean: pd.Series, std: pd.Series) -> dict:
    return score_raw(inverse_ot(pred_scaled, mean, std), inverse_ot(true_scaled, mean, std))


@torch.no_grad()
def predict_seq_residual_booster(model: nn.Module, seq_features: np.ndarray, aux_features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(seq_features), BATCH_SIZE):
        seq_x = torch.from_numpy(seq_features[i : i + BATCH_SIZE]).to(device=device, dtype=torch.float32)
        aux_x = torch.from_numpy(aux_features[i : i + BATCH_SIZE]).to(device=device, dtype=torch.float32)
        preds.append(model(seq_x, aux_x).cpu().numpy())
    return np.concatenate(preds).astype("float32")


def score_residual_lambdas(
    residual_pred: np.ndarray,
    baseline_scaled: np.ndarray,
    residual_true: np.ndarray,
    mean: pd.Series,
    std: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    true_scaled = baseline_scaled + residual_true
    rows = []
    best_metric = None
    best_mse = float("inf")
    for lambda_value in RESIDUAL_LAMBDAS:
        pred_scaled = baseline_scaled + float(lambda_value) * residual_pred
        m = metric(pred_scaled, true_scaled, mean, std)
        rows.append({"lambda": float(lambda_value), **{k: m[k] for k in ["mse", "rmse", "mae"]}})
        if m["mse"] < best_mse:
            best_mse = m["mse"]
            best_metric = {"lambda": float(lambda_value), **m}
    assert best_metric is not None
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True), best_metric


def evaluate_seq_residual_booster_on_targets(
    model: nn.Module,
    data: dict,
    baseline_predictor: dict,
    lambda_value: float,
    target_indices: np.ndarray,
    device: torch.device,
) -> dict:
    seq_x, aux_x, residual_true, baseline_scaled = build_seq_residual_arrays(data, baseline_predictor, target_indices)
    residual_pred = predict_seq_residual_booster(model, seq_x, aux_x, device)
    true_scaled = baseline_scaled + residual_true
    pred_scaled = baseline_scaled + float(lambda_value) * residual_pred
    return metric(pred_scaled, true_scaled, data["mean"], data["std"])


def make_grad_scaler(device: torch.device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
        except TypeError:
            return torch.amp.GradScaler(enabled=(device.type == "cuda"))
    return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def evaluate_seq_residual_validation(
    model: nn.Module,
    val_seq_x: np.ndarray,
    val_aux_x: np.ndarray,
    val_residual: np.ndarray,
    val_baseline_scaled: np.ndarray,
    mean: pd.Series,
    std: pd.Series,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    residual_pred = predict_seq_residual_booster(model, val_seq_x, val_aux_x, device)
    return score_residual_lambdas(residual_pred, val_baseline_scaled, val_residual, mean, std)


def train_seq_residual_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    label: str,
    phase: str,
    epoch: int,
) -> float:
    model.train()
    losses = []
    start_time = time.time()
    total_batches = len(loader)

    for batch_idx, (seq_x, aux_x, yb) in enumerate(loader, start=1):
        seq_x = seq_x.to(device)
        aux_x = aux_x.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(seq_x, aux_x)
            loss = criterion(pred, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

        if batch_idx == 1 or batch_idx % LOG_INTERVAL == 0 or batch_idx == total_batches:
            elapsed = time.time() - start_time
            eta = elapsed / batch_idx * (total_batches - batch_idx)
            recent = np.mean(losses[-LOG_INTERVAL:])
            print(
                f"[{label} {phase} {epoch:03d}] batch {batch_idx:04d}/{total_batches:04d} "
                f"loss={recent:.5f} eta={eta:.1f}s",
                flush=True,
            )

    return float(np.mean(losses))


def train_seq_residual_booster(
    data: dict,
    baseline_predictor: dict,
    device: torch.device,
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
    save_path: str | None = RESIDUAL_MODEL_PATH,
    label: str = "SeqResidualBooster",
) -> tuple[nn.Module, pd.DataFrame, dict]:
    pretrain_seq_x, pretrain_aux_x, pretrain_residual, _ = build_seq_residual_arrays(data, baseline_predictor, data["pretrain_targets"])
    train_seq_x, train_aux_x, train_residual, _ = build_seq_residual_arrays(data, baseline_predictor, data["seq_train_targets"])
    val_seq_x, val_aux_x, val_residual, val_baseline_scaled = build_seq_residual_arrays(data, baseline_predictor, data["val_targets"])

    pretrain_loader = DataLoader(
        SeqResidualDataset(pretrain_seq_x, pretrain_aux_x, pretrain_residual),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    train_loader = DataLoader(
        SeqResidualDataset(train_seq_x, train_aux_x, train_residual),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model = SeqResidualBooster(seq_input_dim=train_seq_x.shape[-1], aux_input_dim=train_aux_x.shape[-1]).to(device)
    print(f"{label} pretrain seq/aux/y:", pretrain_seq_x.shape, pretrain_aux_x.shape, pretrain_residual.shape)
    print(f"{label} finetune seq/aux/y:", train_seq_x.shape, train_aux_x.shape, train_residual.shape)
    print(f"{label} val seq/aux/y:", val_seq_x.shape, val_aux_x.shape, val_residual.shape)
    print(f"{label} params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    scaler = make_grad_scaler(device)
    amp_enabled = device.type == "cuda"

    lambda_df, best_metric = evaluate_seq_residual_validation(
        model,
        val_seq_x,
        val_aux_x,
        val_residual,
        val_baseline_scaled,
        data["mean"],
        data["std"],
        device,
    )
    baseline_metric = metric(val_baseline_scaled, val_baseline_scaled + val_residual, data["mean"], data["std"])
    assert abs(best_metric["mse"] - baseline_metric["mse"]) < 1e-4
    best_mse = best_metric["mse"]
    best_lambda = best_metric["lambda"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_training_plan = {"pretrain_epochs": 0, "finetune_epochs": 0}
    history = [
        {
            "epoch": 0,
            "phase": "init",
            "phase_epoch": 0,
            "train_loss": np.nan,
            **{k: best_metric[k] for k in ["mse", "rmse", "mae"]},
            "lambda": best_lambda,
        }
    ]
    print(f"{label} lambda validation")
    print(lambda_df)
    print(f"[{label} 000] val_mse={best_mse:.5f} lambda={best_lambda:.2f} target_met={best_mse < MSE_TARGET}")
    if save_path is not None:
        torch.save(
            {
                "model_state": best_state,
                "best_mse": best_mse,
                "lambda": best_lambda,
                "training_plan": best_training_plan,
            },
            save_path,
        )

    global_epoch = 0
    for phase, loader, phase_epochs in [
        ("pretrain", pretrain_loader, pretrain_epochs),
        ("finetune", train_loader, finetune_epochs),
    ]:
        bad_epochs = 0
        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            train_loss = train_seq_residual_epoch(
                model,
                loader,
                optimizer,
                criterion,
                scaler,
                device,
                amp_enabled,
                label,
                phase,
                phase_epoch,
            )
            lambda_df, epoch_metric = evaluate_seq_residual_validation(
                model,
                val_seq_x,
                val_aux_x,
                val_residual,
                val_baseline_scaled,
                data["mean"],
                data["std"],
                device,
            )
            scheduler.step(epoch_metric["mse"])
            history.append(
                {
                    "epoch": global_epoch,
                    "phase": phase,
                    "phase_epoch": phase_epoch,
                    "train_loss": train_loss,
                    **{k: epoch_metric[k] for k in ["mse", "rmse", "mae"]},
                    "lambda": epoch_metric["lambda"],
                }
            )
            print(
                f"[{label} {phase} {phase_epoch:03d}] train_loss={train_loss:.5f} "
                f"val_mse={epoch_metric['mse']:.5f} val_rmse={epoch_metric['rmse']:.5f} "
                f"val_mae={epoch_metric['mae']:.5f} lambda={epoch_metric['lambda']:.2f} "
                f"target_met={epoch_metric['mse'] < MSE_TARGET}"
            )

            if epoch_metric["mse"] < best_mse:
                best_mse = epoch_metric["mse"]
                best_lambda = epoch_metric["lambda"]
                best_metric = epoch_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_training_plan = {
                    "pretrain_epochs": phase_epoch if phase == "pretrain" else pretrain_epochs,
                    "finetune_epochs": phase_epoch if phase == "finetune" else 0,
                }
                if save_path is not None:
                    torch.save(
                        {
                            "model_state": best_state,
                            "best_mse": best_mse,
                            "lambda": best_lambda,
                            "training_plan": best_training_plan,
                        },
                        save_path,
                    )
                bad_epochs = 0
            else:
                bad_epochs += 1
                if phase == "finetune" and bad_epochs >= PATIENCE:
                    print(f"{label} finetune early stopping")
                    break

    model.load_state_dict(best_state)
    final_lambda_df, final_metric = evaluate_seq_residual_validation(
        model,
        val_seq_x,
        val_aux_x,
        val_residual,
        val_baseline_scaled,
        data["mean"],
        data["std"],
        device,
    )
    print(f"best {label} val MSE:", final_metric["mse"], "lambda:", final_metric["lambda"], "target met:", final_metric["mse"] < MSE_TARGET)
    print(final_lambda_df)
    if final_metric["mse"] >= MSE_TARGET:
        print(f"{label} warning: validation MSE target not met ({final_metric['mse']:.5f} >= {MSE_TARGET}); baseline fallback can still be selected.")
    print(f"{label} selected training plan:", best_training_plan)
    return model, pd.DataFrame(history), {
        "metric": final_metric,
        "lambda": final_metric["lambda"],
        "lambda_df": final_lambda_df,
        "training_plan": best_training_plan,
    }


def fit_seq_residual_booster_refit(
    data: dict,
    baseline_predictor: dict,
    device: torch.device,
    training_plan: dict,
    label: str,
) -> tuple[nn.Module, pd.DataFrame]:
    pretrain_seq_x, pretrain_aux_x, pretrain_residual, _ = build_seq_residual_arrays(data, baseline_predictor, data["pretrain_targets"])
    train_seq_x, train_aux_x, train_residual, _ = build_seq_residual_arrays(data, baseline_predictor, data["seq_train_targets"])
    pretrain_loader = DataLoader(
        SeqResidualDataset(pretrain_seq_x, pretrain_aux_x, pretrain_residual),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    train_loader = DataLoader(
        SeqResidualDataset(train_seq_x, train_aux_x, train_residual),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    set_seed()
    model = SeqResidualBooster(seq_input_dim=train_seq_x.shape[-1], aux_input_dim=train_aux_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scaler = make_grad_scaler(device)
    amp_enabled = device.type == "cuda"
    history = []

    print(f"{label} pretrain seq/aux/y:", pretrain_seq_x.shape, pretrain_aux_x.shape, pretrain_residual.shape)
    print(f"{label} finetune seq/aux/y:", train_seq_x.shape, train_aux_x.shape, train_residual.shape)
    print(f"{label} training plan:", training_plan)
    for phase, loader, phase_epochs in [
        ("refit-pretrain", pretrain_loader, int(training_plan["pretrain_epochs"])),
        ("refit-finetune", train_loader, int(training_plan["finetune_epochs"])),
    ]:
        for phase_epoch in range(1, phase_epochs + 1):
            train_loss = train_seq_residual_epoch(
                model,
                loader,
                optimizer,
                criterion,
                scaler,
                device,
                amp_enabled,
                label,
                phase,
                phase_epoch,
            )
            history.append({"phase": phase, "phase_epoch": phase_epoch, "train_loss": train_loss})

    return model, pd.DataFrame(history, columns=["phase", "phase_epoch", "train_loss"])


@torch.no_grad()
def predict_patchtst_residual(model: nn.Module, dataset: PatchResidualDataset, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=PATCH_BATCH_SIZE, shuffle=False)
    preds = []
    for seq_x, aux_x, _ in loader:
        seq_x = seq_x.to(device)
        aux_x = aux_x.to(device)
        preds.append(model(seq_x, aux_x).cpu().numpy())
    return np.concatenate(preds).astype("float32")


def evaluate_patchtst_validation(
    model: nn.Module,
    val_dataset: PatchResidualDataset,
    mean: pd.Series,
    std: pd.Series,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    residual_pred = predict_patchtst_residual(model, val_dataset, device)
    return score_residual_lambdas(residual_pred, val_dataset.baseline_scaled, val_dataset.residual, mean, std)


def train_patchtst_residual(
    data: dict,
    baseline_predictor: dict,
    device: torch.device,
    pretrain_epochs: int = PATCH_PRETRAIN_EPOCHS,
    finetune_epochs: int = PATCH_FINETUNE_EPOCHS,
    save_path: str | None = PATCH_MODEL_PATH,
    label: str = "PatchTSTResidual",
) -> tuple[nn.Module, pd.DataFrame, dict]:
    pretrain_ds = PatchResidualDataset(data, baseline_predictor, data["patch_pretrain_targets"])
    train_ds = PatchResidualDataset(data, baseline_predictor, data["patch_train_targets"])
    val_ds = PatchResidualDataset(data, baseline_predictor, data["val_targets"])

    sample_seq, sample_aux, _ = train_ds[0]
    model = PatchTSTResidual(seq_input_dim=sample_seq.shape[-1], aux_input_dim=sample_aux.shape[-1]).to(device)
    print(f"{label} pretrain/finetune/val:", len(pretrain_ds), len(train_ds), len(val_ds))
    print(f"{label} sample seq/aux/y:", sample_seq.shape, sample_aux.shape, train_ds.residual.shape)
    print(f"{label} params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    pretrain_loader = DataLoader(pretrain_ds, batch_size=PATCH_BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=PATCH_BATCH_SIZE, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    scaler = make_grad_scaler(device)
    amp_enabled = device.type == "cuda"

    lambda_df, best_metric = evaluate_patchtst_validation(model, val_ds, data["mean"], data["std"], device)
    baseline_metric = metric(val_ds.baseline_scaled, val_ds.baseline_scaled + val_ds.residual, data["mean"], data["std"])
    assert abs(best_metric["mse"] - baseline_metric["mse"]) < 1e-4
    best_mse = best_metric["mse"]
    best_lambda = best_metric["lambda"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, "phase": "init", "train_loss": np.nan, **{k: best_metric[k] for k in ["mse", "rmse", "mae"]}, "lambda": best_lambda}]
    print(f"{label} lambda validation")
    print(lambda_df)
    print(f"[{label} 000] val_mse={best_mse:.5f} lambda={best_lambda:.2f} target_met={best_mse < MSE_TARGET}")
    if save_path is not None:
        torch.save({"model_state": best_state, "best_mse": best_mse, "lambda": best_lambda}, save_path)

    global_epoch = 0
    for phase, loader, phase_epochs in [
        ("pretrain", pretrain_loader, pretrain_epochs),
        ("finetune", train_loader, finetune_epochs),
    ]:
        bad_epochs = 0
        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            train_loss = train_seq_residual_epoch(
                model,
                loader,
                optimizer,
                criterion,
                scaler,
                device,
                amp_enabled,
                label,
                phase,
                phase_epoch,
            )
            lambda_df, epoch_metric = evaluate_patchtst_validation(model, val_ds, data["mean"], data["std"], device)
            scheduler.step(epoch_metric["mse"])
            history.append(
                {
                    "epoch": global_epoch,
                    "phase": phase,
                    "train_loss": train_loss,
                    **{k: epoch_metric[k] for k in ["mse", "rmse", "mae"]},
                    "lambda": epoch_metric["lambda"],
                }
            )
            print(
                f"[{label} {phase} {phase_epoch:03d}] train_loss={train_loss:.5f} "
                f"val_mse={epoch_metric['mse']:.5f} val_rmse={epoch_metric['rmse']:.5f} "
                f"val_mae={epoch_metric['mae']:.5f} lambda={epoch_metric['lambda']:.2f} "
                f"target_met={epoch_metric['mse'] < MSE_TARGET}"
            )

            if epoch_metric["mse"] < best_mse:
                best_mse = epoch_metric["mse"]
                best_lambda = epoch_metric["lambda"]
                best_metric = epoch_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if save_path is not None:
                    torch.save({"model_state": best_state, "best_mse": best_mse, "lambda": best_lambda}, save_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if phase == "finetune" and bad_epochs >= PATCH_PATIENCE:
                    print(f"{label} finetune early stopping")
                    break

    model.load_state_dict(best_state)
    final_lambda_df, final_metric = evaluate_patchtst_validation(model, val_ds, data["mean"], data["std"], device)
    print(f"best {label} val MSE:", final_metric["mse"], "lambda:", final_metric["lambda"], "target met:", final_metric["mse"] < MSE_TARGET)
    print(final_lambda_df)
    return model, pd.DataFrame(history), {"metric": final_metric, "lambda": final_metric["lambda"], "lambda_df": final_lambda_df}


def evaluate_patchtst_residual_on_targets(
    model: nn.Module,
    data: dict,
    baseline_predictor: dict,
    lambda_value: float,
    target_indices: np.ndarray,
    device: torch.device,
) -> dict:
    dataset = PatchResidualDataset(data, baseline_predictor, target_indices)
    residual_pred = predict_patchtst_residual(model, dataset, device)
    true_scaled = dataset.baseline_scaled + dataset.residual
    pred_scaled = dataset.baseline_scaled + float(lambda_value) * residual_pred
    return metric(pred_scaled, true_scaled, data["mean"], data["std"])


def rolling_patchtst_residual_backtest(data: dict, device: torch.device) -> pd.DataFrame:
    df = data["df"]
    train_end_idx = data["train_end_idx"]
    fold_starts = [
        train_end_idx - 24 * 90,
        train_end_idx - 24 * 45,
        train_end_idx,
    ]
    rows = []

    for fold_idx, outer_start in enumerate(fold_starts, start=1):
        outer_end = outer_start + 24 * 142
        if outer_end > data["test_start_idx"]:
            continue

        inner_train_end = int(outer_start * TRAIN_FRACTION_BEFORE_TEST)
        inner_train_end = (inner_train_end // 24) * 24
        inner_train_targets = midnight_target_indices(df, LOOKBACK, inner_train_end)
        inner_val_targets = midnight_target_indices(df, inner_train_end, outer_start)
        outer_targets = midnight_target_indices(df, outer_start, outer_end)

        if len(inner_train_targets) == 0 or len(inner_val_targets) == 0 or len(outer_targets) == 0:
            continue

        fold_data = make_patchtst_data_view(data, inner_train_end, inner_train_targets, inner_val_targets)
        _, fold_baseline = fit_baseline_candidates(fold_data)
        model, _, residual_info = train_patchtst_residual(
            fold_data,
            fold_baseline,
            device,
            pretrain_epochs=max(1, PATCH_BACKTEST_EPOCHS // 2),
            finetune_epochs=PATCH_BACKTEST_EPOCHS,
            save_path=None,
            label=f"PatchTSTBT{fold_idx}",
        )

        baseline_outer = evaluate_baseline_on_targets(fold_data, fold_baseline, outer_targets)
        residual_outer = evaluate_patchtst_residual_on_targets(
            model,
            fold_data,
            fold_baseline,
            residual_info["lambda"],
            outer_targets,
            device,
        )
        selected_outer = residual_outer if residual_info["lambda"] > 0.0 else baseline_outer

        rows.append(
            {
                "fold": fold_idx,
                "outer_start": df.loc[outer_targets[0], "date"],
                "outer_end": df.loc[outer_targets[-1] + HORIZON - 1, "date"],
                "baseline": fold_baseline["name"],
                "inner_baseline_mse": fold_baseline["metric"]["mse"],
                "inner_residual_mse": residual_info["metric"]["mse"],
                "lambda": residual_info["lambda"],
                "outer_baseline_mse": baseline_outer["mse"],
                "outer_residual_mse": residual_outer["mse"],
                "outer_selected_mse": selected_outer["mse"],
                "outer_selected_mae": selected_outer["mae"],
                "improvement": baseline_outer["mse"] - selected_outer["mse"],
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    summary = {
        "folds": len(out),
        "baseline_mean_mse": out["outer_baseline_mse"].mean(),
        "residual_mean_mse": out["outer_residual_mse"].mean(),
        "selected_mean_mse": out["outer_selected_mse"].mean(),
        "baseline_max_mse": out["outer_baseline_mse"].max(),
        "selected_max_mse": out["outer_selected_mse"].max(),
        "mean_improvement": out["improvement"].mean(),
    }
    print("rolling PatchTST residual backtest")
    print(out)
    print("rolling PatchTST residual summary")
    print(pd.DataFrame([summary]))
    return out


def rolling_residual_backtest(data: dict, device: torch.device) -> pd.DataFrame:
    df = data["df"]
    train_end_idx = data["train_end_idx"]
    fold_starts = [
        train_end_idx - 24 * 90,
        train_end_idx - 24 * 45,
        train_end_idx,
    ]
    rows = []

    for fold_idx, outer_start in enumerate(fold_starts, start=1):
        outer_end = outer_start + 24 * 142
        if outer_end > data["test_start_idx"]:
            continue

        inner_train_end = int(outer_start * TRAIN_FRACTION_BEFORE_TEST)
        inner_train_end = (inner_train_end // 24) * 24
        inner_train_targets = midnight_target_indices(df, LOOKBACK, inner_train_end)
        inner_val_targets = midnight_target_indices(df, inner_train_end, outer_start)
        outer_targets = midnight_target_indices(df, outer_start, outer_end)

        if len(inner_train_targets) == 0 or len(inner_val_targets) == 0 or len(outer_targets) == 0:
            continue

        fold_data = make_data_view(data, inner_train_end, inner_train_targets, inner_val_targets)
        _, fold_baseline = fit_baseline_candidates(fold_data)
        model, _, residual_info = train_seq_residual_booster(
            fold_data,
            fold_baseline,
            device,
            pretrain_epochs=max(1, RESIDUAL_BACKTEST_EPOCHS // 2),
            finetune_epochs=RESIDUAL_BACKTEST_EPOCHS,
            save_path=None,
            label=f"SeqResidualBT{fold_idx}",
        )

        baseline_outer = evaluate_baseline_on_targets(fold_data, fold_baseline, outer_targets)
        validation_stage_outer = evaluate_seq_residual_booster_on_targets(
            model,
            fold_data,
            fold_baseline,
            residual_info["lambda"],
            outer_targets,
            device,
        )
        refit_data = make_data_view(
            data,
            outer_start,
            midnight_target_indices(df, LOOKBACK, outer_start),
            outer_targets,
        )
        refit_model, _ = fit_seq_residual_booster_refit(
            refit_data,
            fold_baseline,
            device,
            residual_info["training_plan"],
            label=f"SeqResidualBT{fold_idx}Refit",
        )
        residual_outer = evaluate_seq_residual_booster_on_targets(
            refit_model,
            refit_data,
            fold_baseline,
            residual_info["lambda"],
            outer_targets,
            device,
        )
        selected_outer = residual_outer if residual_info["lambda"] > 0.0 else baseline_outer

        rows.append(
            {
                "fold": fold_idx,
                "outer_start": df.loc[outer_targets[0], "date"],
                "outer_end": df.loc[outer_targets[-1] + HORIZON - 1, "date"],
                "baseline": fold_baseline["name"],
                "inner_baseline_mse": fold_baseline["metric"]["mse"],
                "inner_residual_mse": residual_info["metric"]["mse"],
                "lambda": residual_info["lambda"],
                "refit_pretrain_epochs": residual_info["training_plan"]["pretrain_epochs"],
                "refit_finetune_epochs": residual_info["training_plan"]["finetune_epochs"],
                "outer_baseline_mse": baseline_outer["mse"],
                "outer_validation_stage_residual_mse": validation_stage_outer["mse"],
                "outer_residual_mse": residual_outer["mse"],
                "outer_selected_mse": selected_outer["mse"],
                "outer_selected_mae": selected_outer["mae"],
                "improvement": baseline_outer["mse"] - selected_outer["mse"],
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    summary = {
        "folds": len(out),
        "baseline_mean_mse": out["outer_baseline_mse"].mean(),
        "residual_mean_mse": out["outer_residual_mse"].mean(),
        "selected_mean_mse": out["outer_selected_mse"].mean(),
        "baseline_max_mse": out["outer_baseline_mse"].max(),
        "selected_max_mse": out["outer_selected_mse"].max(),
        "mean_improvement": out["improvement"].mean(),
    }
    print("rolling residual backtest")
    print(out)
    print("rolling residual summary")
    print(pd.DataFrame([summary]))
    return out


def train_model(data: dict, device: torch.device) -> tuple[nn.Module, list[dict], dict]:
    train_loader = DataLoader(
        WindowDataset(data["values_scaled"], data["target_scaled"], data["train_targets"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        WindowDataset(data["values_scaled"], data["target_scaled"], data["val_targets"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    xb, yb = next(iter(train_loader))
    print("batch X:", xb.shape, "batch y:", yb.shape)

    model = GRUForecast(input_dim=len(FEATURE_COLS)).to(device)
    print("params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = make_grad_scaler(device)
    amp_enabled = device.type == "cuda"

    pred_val, true_val = predict_loader(model, val_loader, device)
    init_metric = metric(pred_val, true_val, data["mean"], data["std"])
    best_mse = init_metric["mse"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    history = [{"epoch": 0, "train_loss": np.nan, **{k: init_metric[k] for k in ["mse", "rmse", "mae"]}}]
    print(
        f"[Epoch 000] residual baseline val_mse={best_mse:.5f} "
        f"val_rmse={init_metric['rmse']:.5f} target_met={best_mse < MSE_TARGET}"
    )
    torch.save({"model_state": best_state, "best_mse": best_mse}, BEST_MODEL_PATH)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        start_time = time.time()
        total_batches = len(train_loader)

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

            if batch_idx == 1 or batch_idx % LOG_INTERVAL == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_time
                eta = elapsed / batch_idx * (total_batches - batch_idx)
                recent = np.mean(losses[-LOG_INTERVAL:])
                print(
                    f"[Epoch {epoch:03d}] batch {batch_idx:04d}/{total_batches:04d} "
                    f"loss={recent:.5f} eta={eta:.1f}s",
                    flush=True,
                )

        pred_val, true_val = predict_loader(model, val_loader, device)
        m = metric(pred_val, true_val, data["mean"], data["std"])
        scheduler.step(m["mse"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{k: m[k] for k in ["mse", "rmse", "mae"]}})
        print(
            f"[Epoch {epoch:03d}] train_loss={np.mean(losses):.5f} "
            f"val_mse={m['mse']:.5f} val_rmse={m['rmse']:.5f} "
            f"val_mae={m['mae']:.5f} target_met={m['mse'] < MSE_TARGET}"
        )

        if m["mse"] < best_mse:
            best_mse = m["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state": best_state, "best_mse": best_mse}, BEST_MODEL_PATH)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    print("best GRU val MSE:", best_mse, "target met:", best_mse < MSE_TARGET)
    assert best_mse < MSE_TARGET, f"validation MSE target not met: {best_mse:.5f} >= {MSE_TARGET}"
    return model, history, {"best_mse": best_mse, "val_loader": val_loader}


def residual_backtest_passes(backtest: pd.DataFrame) -> tuple[bool, str]:
    if backtest.empty:
        return True, "no residual backtest gate"

    improvements = backtest["improvement"].astype(float)
    mean_improvement = float(improvements.mean())
    positive_folds = int((improvements > 0.0).sum())
    required_folds = max(1, len(backtest) // 2 + 1)
    baseline_max = float(backtest["outer_baseline_mse"].max())
    selected_max = float(backtest["outer_selected_mse"].max())

    passes = mean_improvement > 0.0 and positive_folds >= required_folds and selected_max <= baseline_max * 1.05
    reason = (
        f"residual backtest mean_improvement={mean_improvement:.6f}, "
        f"positive_folds={positive_folds}/{len(backtest)}, "
        f"selected_max={selected_max:.6f}, baseline_max={baseline_max:.6f}"
    )
    return passes, reason


def choose_predictor(best_baseline: dict, residual_info: dict, residual_backtest: pd.DataFrame | None = None) -> dict:
    residual_metric = residual_info["metric"]
    if residual_metric["lambda"] <= 0.0 or residual_metric["mse"] >= best_baseline["metric"]["mse"]:
        selected = dict(best_baseline)
        selected["selection_reason"] = "baseline selected because residual did not improve validation MSE"
        return selected

    if residual_backtest is not None:
        backtest_ok, backtest_reason = residual_backtest_passes(residual_backtest)
        if not backtest_ok:
            selected = dict(best_baseline)
            selected["selection_reason"] = f"baseline selected because residual failed backtest gate: {backtest_reason}"
            return selected

        return {
            "name": f"seq_residual_booster_{best_baseline['name']}_lambda{residual_metric['lambda']:.2f}",
            "kind": "seq_residual_booster",
            "baseline": best_baseline,
            "lambda": residual_metric["lambda"],
            "metric": residual_metric,
            "selection_reason": "residual selected by validation MSE and residual backtest gate",
        }

    return {
        "name": f"seq_residual_booster_{best_baseline['name']}_lambda{residual_metric['lambda']:.2f}",
        "kind": "seq_residual_booster",
        "baseline": best_baseline,
        "lambda": residual_metric["lambda"],
        "metric": residual_metric,
        "selection_reason": "residual selected by validation MSE; residual backtest gate not run",
    }


def choose_patchtst_predictor(best_baseline: dict, residual_info: dict, residual_backtest: pd.DataFrame | None = None) -> dict:
    residual_metric = residual_info["metric"]
    if residual_metric["lambda"] <= 0.0 or residual_metric["mse"] >= best_baseline["metric"]["mse"]:
        selected = dict(best_baseline)
        selected["selection_reason"] = "baseline selected because PatchTST residual did not improve validation MSE"
        return selected

    if residual_backtest is not None:
        backtest_ok, backtest_reason = residual_backtest_passes(residual_backtest)
        if not backtest_ok:
            selected = dict(best_baseline)
            selected["selection_reason"] = f"baseline selected because PatchTST residual failed backtest gate: {backtest_reason}"
            return selected

        return {
            "name": f"patchtst_residual_{best_baseline['name']}_lambda{residual_metric['lambda']:.2f}",
            "kind": "patchtst_residual",
            "baseline": best_baseline,
            "lambda": residual_metric["lambda"],
            "metric": residual_metric,
            "selection_reason": "PatchTST residual selected by validation MSE and residual backtest gate",
        }

    return {
        "name": f"patchtst_residual_{best_baseline['name']}_lambda{residual_metric['lambda']:.2f}",
        "kind": "patchtst_residual",
        "baseline": best_baseline,
        "lambda": residual_metric["lambda"],
        "metric": residual_metric,
        "selection_reason": "PatchTST residual selected by validation MSE; residual backtest gate not run",
    }


@torch.no_grad()
def make_submission(
    predictor: dict,
    model: nn.Module,
    df: pd.DataFrame,
    sub: pd.DataFrame,
    id_col: str,
    data: dict,
    device: torch.device,
) -> pd.DataFrame:
    date_to_idx = {d: i for i, d in enumerate(df["date"])}
    target_indices = []
    for ts in sub[id_col]:
        target_ts = pd.Timestamp(ts)
        target_idx = date_to_idx[target_ts]
        max_input_idx = target_idx - 1
        assert max_input_idx >= 0
        assert df.loc[max_input_idx, "date"] < target_ts
        if predictor["kind"] == "patchtst_residual":
            assert target_idx - PATCH_START_LOOKBACK >= 0
        elif predictor["kind"] == "seq_residual_booster":
            assert target_idx - SEQ_LOOKBACK >= 0
        else:
            assert target_idx - LOOKBACK >= 0
        target_indices.append(target_idx)
    target_indices = np.asarray(target_indices, dtype=np.int64)

    if predictor["kind"] == "patchtst_residual":
        dataset = PatchResidualDataset(data, predictor["baseline"], target_indices, include_target=False)
        residual_scaled = predict_patchtst_residual(model, dataset, device)
        pred_scaled = dataset.baseline_scaled + float(predictor["lambda"]) * residual_scaled
        pred_values = inverse_ot(pred_scaled, data["mean"], data["std"])
    elif predictor["kind"] == "seq_residual_booster":
        seq_x, aux_x, _, baseline_scaled = build_seq_residual_arrays(data, predictor["baseline"], target_indices, include_target=False)
        residual_scaled = predict_seq_residual_booster(model, seq_x, aux_x, device)
        pred_scaled = baseline_scaled + float(predictor["lambda"]) * residual_scaled
        pred_values = inverse_ot(pred_scaled, data["mean"], data["std"])
    else:
        raw_ot = df[TARGET_COL].values.astype("float32")
        pred_values = baseline_prediction(raw_ot, target_indices, predictor)

    out = sub.copy()
    t_cols = [f"T{i}" for i in range(HORIZON)]
    out[t_cols] = pred_values
    out[id_col] = pd.to_datetime(out[id_col]).dt.strftime("%Y-%m-%d")
    out = out[[id_col] + t_cols]
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("selected predictor:", predictor["name"])
    print("saved:", OUTPUT_PATH, out.shape)
    print(out.head())
    print(out.tail())
    print(out[t_cols].describe().loc[["mean", "std", "min", "max"]])

    assert out.shape == sub[[id_col] + t_cols].shape
    assert np.isfinite(out[t_cols].values).all()
    return out


def plot_history(history: list[dict], final_metric: dict) -> None:
    import matplotlib.pyplot as plt

    hist = pd.DataFrame(history)
    print(hist.tail())

    plt.figure(figsize=(8, 4))
    plt.plot(hist["epoch"], hist["train_loss"], label="train scaled MSE")
    plt.plot(hist["epoch"], hist["mse"], label="DL val raw MSE")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(final_metric["horizon_rmse"])
    plt.title("Selected validation RMSE by horizon")
    plt.xlabel("horizon")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()


def run(plot: bool = True, residual_backtest: bool = True) -> dict:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df, sub, id_col = load_data()
    data = prepare_data(df)

    baseline_df, best_baseline = fit_baseline_candidates(data)
    print(baseline_df)
    print("best baseline:", best_baseline["name"], best_baseline["metric"]["mse"], "target met:", best_baseline["metric"]["mse"] < MSE_TARGET)
    backtest_df = rolling_backtest(data)

    model, history, residual_info = train_seq_residual_booster(data, best_baseline, device)
    print("Seq residual booster validation:", {k: v for k, v in residual_info["metric"].items() if k != "horizon_rmse"})

    residual_backtest_df = rolling_residual_backtest(data, device) if residual_backtest else pd.DataFrame()

    selected = choose_predictor(best_baseline, residual_info, residual_backtest_df if residual_backtest else None)
    final_metric = selected["metric"]
    print("selected predictor:", selected["name"])
    print("selection reason:", selected.get("selection_reason", ""))
    print("best_mse:", final_metric["mse"], "target_met:", final_metric["mse"] < MSE_TARGET)
    assert final_metric["mse"] <= float(baseline_df.loc[baseline_df["name"] == "last_value", "mse"].iloc[0])
    if final_metric["mse"] >= MSE_TARGET:
        print(f"warning: selected validation MSE target not met ({final_metric['mse']:.5f} >= {MSE_TARGET}); continuing to write submission.")

    if plot:
        plot_history(history, final_metric)

    submission_model = model
    submission_data = data
    refit_history = pd.DataFrame(columns=["phase", "phase_epoch", "train_loss"])
    if selected["kind"] == "seq_residual_booster":
        submission_data = make_data_view(
            data,
            data["test_start_idx"],
            midnight_target_indices(data["df"], LOOKBACK, data["test_start_idx"]),
            data["val_targets"],
        )
        submission_model, refit_history = fit_seq_residual_booster_refit(
            submission_data,
            best_baseline,
            device,
            residual_info["training_plan"],
            label="SeqResidualFinalRefit",
        )
    submit = make_submission(selected, submission_model, submission_data["df"], sub, id_col, submission_data, device)
    return {
        "model": submission_model,
        "history": history,
        "refit_history": refit_history,
        "baseline": baseline_df,
        "metric": final_metric,
        "predictor": selected,
        "backtest": backtest_df,
        "residual_backtest": residual_backtest_df,
        "residual_lambda": residual_info["lambda_df"],
        "residual_training_plan": residual_info["training_plan"],
        "submit": submit,
    }


def run_patchtst(plot: bool = True, residual_backtest: bool = True) -> dict:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df, sub, id_col = load_patchtst_data()
    data = prepare_patchtst_data(df)

    baseline_df, best_baseline = fit_baseline_candidates(data)
    print(baseline_df)
    print("best baseline:", best_baseline["name"], best_baseline["metric"]["mse"], "target met:", best_baseline["metric"]["mse"] < MSE_TARGET)
    backtest_df = rolling_backtest(data)

    model, history, residual_info = train_patchtst_residual(data, best_baseline, device)
    print("PatchTST residual validation:", {k: v for k, v in residual_info["metric"].items() if k != "horizon_rmse"})

    residual_backtest_df = rolling_patchtst_residual_backtest(data, device) if residual_backtest else pd.DataFrame()

    selected = choose_patchtst_predictor(best_baseline, residual_info, residual_backtest_df if residual_backtest else None)
    final_metric = selected["metric"]
    print("selected predictor:", selected["name"])
    print("selection reason:", selected.get("selection_reason", ""))
    print("best_mse:", final_metric["mse"], "target_met:", final_metric["mse"] < MSE_TARGET)
    assert final_metric["mse"] <= float(baseline_df.loc[baseline_df["name"] == "last_value", "mse"].iloc[0])
    if final_metric["mse"] > 4.90:
        print(f"warning: selected validation MSE {final_metric['mse']:.5f} is above the rough public-6.00 target proxy of 4.90.")

    if plot:
        plot_history(history, final_metric)

    submit = make_submission(selected, model, data["df"], sub, id_col, data, device)
    return {
        "model": model,
        "history": history,
        "baseline": baseline_df,
        "metric": final_metric,
        "predictor": selected,
        "backtest": backtest_df,
        "residual_backtest": residual_backtest_df,
        "residual_lambda": residual_info["lambda_df"],
        "submit": submit,
    }


if __name__ == "__main__":
    run(plot=True)
