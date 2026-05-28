from __future__ import annotations

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
DATA_PATH = "ETTh1.csv"
SAMPLE_PATH = "sample_submit.csv"
OUTPUT_PATH = "submit.csv"
BEST_MODEL_PATH = "best_model.pt"
RESIDUAL_MODEL_PATH = "best_residual_mlp.pt"
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
LOOKBACK = 168
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
RESIDUAL_HIDDEN_SIZE = 256
RESIDUAL_DROPOUT = 0.35
RESIDUAL_EPOCHS = 80
RESIDUAL_BACKTEST_EPOCHS = 40
RESIDUAL_LAMBDAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


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


def midnight_target_indices(df: pd.DataFrame, start_idx: int, end_idx: int) -> np.ndarray:
    targets = np.arange(start_idx, end_idx - HORIZON + 1)
    targets = targets[df.loc[targets, "date"].dt.hour.values == 0]
    return targets.astype(np.int64)


def prepare_data(df: pd.DataFrame) -> dict:
    df = add_time_features(df)

    test_start_idx = int(np.where(df["date"].values == np.datetime64(TEST_START))[0][0])
    train_end_idx = int(test_start_idx * TRAIN_FRACTION_BEFORE_TEST)
    train_end_idx = (train_end_idx // 24) * 24

    train_targets = midnight_target_indices(df, LOOKBACK, train_end_idx)
    val_targets = midnight_target_indices(df, train_end_idx, test_start_idx)

    print("train target:", df.loc[train_targets[0], "date"], "~", df.loc[train_targets[-1] + HORIZON - 1, "date"])
    print("val target:", df.loc[val_targets[0], "date"], "~", df.loc[val_targets[-1] + HORIZON - 1, "date"])
    print("test starts:", df.loc[test_start_idx, "date"])

    mean = df.loc[: train_end_idx - 1, FEATURE_COLS].mean()
    std = df.loc[: train_end_idx - 1, FEATURE_COLS].std().replace(0, 1.0)
    values_scaled = ((df[FEATURE_COLS] - mean) / std).astype("float32").values
    target_scaled = values_scaled[:, OT_IDX]

    print("feature shape:", values_scaled.shape)
    print("train targets:", len(train_targets), "val targets:", len(val_targets))

    assert len(train_targets) > 0 and len(val_targets) > 0
    assert (df.loc[train_targets, "date"].dt.hour == 0).all()
    assert (df.loc[val_targets, "date"].dt.hour == 0).all()
    assert train_targets[0] - LOOKBACK >= 0
    assert train_targets[-1] + HORIZON <= train_end_idx
    assert val_targets[0] >= train_end_idx
    assert val_targets[-1] + HORIZON <= test_start_idx
    assert (df.loc[val_targets - 1, "date"].values < df.loc[val_targets, "date"].values).all()

    return {
        "df": df,
        "mean": mean,
        "std": std,
        "values_scaled": values_scaled,
        "target_scaled": target_scaled,
        "train_targets": train_targets,
        "val_targets": val_targets,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
    }


def make_data_view(data: dict, train_end_idx: int, train_targets: np.ndarray, val_targets: np.ndarray) -> dict:
    df = data["df"]
    mean = df.loc[: train_end_idx - 1, FEATURE_COLS].mean()
    std = df.loc[: train_end_idx - 1, FEATURE_COLS].std().replace(0, 1.0)
    values_scaled = ((df[FEATURE_COLS] - mean) / std).astype("float32").values

    return {
        "df": df,
        "mean": mean,
        "std": std,
        "values_scaled": values_scaled,
        "target_scaled": values_scaled[:, OT_IDX],
        "train_targets": train_targets,
        "val_targets": val_targets,
        "train_end_idx": train_end_idx,
        "test_start_idx": data["test_start_idx"],
    }


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


class ResidualDataset(Dataset):
    def __init__(self, features: np.ndarray, residual: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.residual = torch.tensor(residual, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.residual[idx]


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


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.out = nn.Linear(RESIDUAL_HIDDEN_SIZE, HORIZON)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, RESIDUAL_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(RESIDUAL_DROPOUT),
            nn.Linear(RESIDUAL_HIDDEN_SIZE, RESIDUAL_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(RESIDUAL_DROPOUT),
            self.out,
        )
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_residual_arrays(
    data: dict,
    baseline_predictor: dict,
    target_indices: np.ndarray,
    include_target: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    df = data["df"]
    values = data["values_scaled"]
    raw_ot = df[TARGET_COL].values.astype("float32")
    ot_mean = float(data["mean"][TARGET_COL])
    ot_std = float(data["std"][TARGET_COL])

    baseline_raw = baseline_prediction(raw_ot, target_indices, baseline_predictor).astype("float32")
    baseline_scaled = ((baseline_raw - ot_mean) / ot_std).astype("float32")

    features = []
    for row_idx, t in enumerate(target_indices):
        recent_24_raw = values[t - 24 : t, RAW_IDXS]
        recent_168_raw = values[t - LOOKBACK : t, RAW_IDXS]
        week_24_raw = values[t - 168 : t - 144, RAW_IDXS]
        future_time = values[t : t + HORIZON, TIME_IDXS].reshape(-1)
        summary = np.concatenate(
            [
                recent_24_raw.mean(axis=0),
                recent_24_raw.std(axis=0),
                recent_168_raw.mean(axis=0),
                recent_168_raw.std(axis=0),
                values[t - 1, RAW_IDXS],
                recent_24_raw.mean(axis=0) - week_24_raw.mean(axis=0),
            ]
        )
        features.append(
            np.concatenate(
                [
                    baseline_scaled[row_idx],
                    values[t - LOOKBACK : t, OT_IDX],
                    recent_24_raw.reshape(-1),
                    summary,
                    future_time,
                ]
            )
        )

    x = np.asarray(features, dtype=np.float32)
    if not include_target:
        return x, None, baseline_scaled

    true_scaled = np.stack([data["target_scaled"][t : t + HORIZON] for t in target_indices]).astype("float32")
    residual = (true_scaled - baseline_scaled).astype("float32")
    return x, residual, baseline_scaled


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
def predict_residual_mlp(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(features), BATCH_SIZE):
        xb = torch.tensor(features[i : i + BATCH_SIZE], dtype=torch.float32, device=device)
        preds.append(model(xb).cpu().numpy())
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


def evaluate_residual_mlp_on_targets(
    model: nn.Module,
    data: dict,
    baseline_predictor: dict,
    lambda_value: float,
    target_indices: np.ndarray,
    device: torch.device,
) -> dict:
    x, residual_true, baseline_scaled = build_residual_arrays(data, baseline_predictor, target_indices)
    residual_pred = predict_residual_mlp(model, x, device)
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


def train_residual_mlp(
    data: dict,
    baseline_predictor: dict,
    device: torch.device,
    epochs: int = RESIDUAL_EPOCHS,
    save_path: str | None = RESIDUAL_MODEL_PATH,
    label: str = "ResidualMLP",
) -> tuple[nn.Module, pd.DataFrame, dict]:
    train_x, train_residual, _ = build_residual_arrays(data, baseline_predictor, data["train_targets"])
    val_x, val_residual, val_baseline_scaled = build_residual_arrays(data, baseline_predictor, data["val_targets"])

    train_loader = DataLoader(
        ResidualDataset(train_x, train_residual),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model = ResidualMLP(input_dim=train_x.shape[1]).to(device)
    print(f"{label} X:", train_x.shape, "residual y:", train_residual.shape)
    print(f"{label} params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 3, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    scaler = make_grad_scaler(device)
    amp_enabled = device.type == "cuda"

    residual_pred = predict_residual_mlp(model, val_x, device)
    lambda_df, best_metric = score_residual_lambdas(residual_pred, val_baseline_scaled, val_residual, data["mean"], data["std"])
    best_mse = best_metric["mse"]
    best_lambda = best_metric["lambda"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    history = [{"epoch": 0, "train_loss": np.nan, **{k: best_metric[k] for k in ["mse", "rmse", "mae"]}, "lambda": best_lambda}]
    print(f"{label} lambda validation")
    print(lambda_df)
    print(f"[{label} 000] val_mse={best_mse:.5f} lambda={best_lambda:.2f} target_met={best_mse < MSE_TARGET}")
    if save_path is not None:
        torch.save({"model_state": best_state, "best_mse": best_mse, "lambda": best_lambda}, save_path)

    for epoch in range(1, epochs + 1):
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
                    f"[{label} {epoch:03d}] batch {batch_idx:04d}/{total_batches:04d} "
                    f"loss={recent:.5f} eta={eta:.1f}s",
                    flush=True,
                )

        residual_pred = predict_residual_mlp(model, val_x, device)
        lambda_df, epoch_metric = score_residual_lambdas(residual_pred, val_baseline_scaled, val_residual, data["mean"], data["std"])
        scheduler.step(epoch_metric["mse"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{k: epoch_metric[k] for k in ["mse", "rmse", "mae"]},
                "lambda": epoch_metric["lambda"],
            }
        )
        print(
            f"[{label} {epoch:03d}] train_loss={np.mean(losses):.5f} "
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
            if bad_epochs >= PATIENCE:
                print(f"{label} early stopping")
                break

    model.load_state_dict(best_state)
    final_residual_pred = predict_residual_mlp(model, val_x, device)
    final_lambda_df, final_metric = score_residual_lambdas(final_residual_pred, val_baseline_scaled, val_residual, data["mean"], data["std"])
    print(f"best {label} val MSE:", final_metric["mse"], "lambda:", final_metric["lambda"], "target met:", final_metric["mse"] < MSE_TARGET)
    print(final_lambda_df)
    assert final_metric["mse"] < MSE_TARGET, f"validation MSE target not met: {final_metric['mse']:.5f} >= {MSE_TARGET}"
    return model, pd.DataFrame(history), {"metric": final_metric, "lambda": final_metric["lambda"], "lambda_df": final_lambda_df}


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
        model, _, residual_info = train_residual_mlp(
            fold_data,
            fold_baseline,
            device,
            epochs=RESIDUAL_BACKTEST_EPOCHS,
            save_path=None,
            label=f"ResidualBT{fold_idx}",
        )

        baseline_outer = evaluate_baseline_on_targets(fold_data, fold_baseline, outer_targets)
        residual_outer = evaluate_residual_mlp_on_targets(
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


def choose_predictor(best_baseline: dict, residual_info: dict) -> dict:
    residual_metric = residual_info["metric"]
    if residual_metric["lambda"] > 0.0 and residual_metric["mse"] < best_baseline["metric"]["mse"]:
        return {
            "name": f"residual_mlp_{best_baseline['name']}_lambda{residual_metric['lambda']:.2f}",
            "kind": "residual_mlp",
            "baseline": best_baseline,
            "lambda": residual_metric["lambda"],
            "metric": residual_metric,
        }
    return best_baseline


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
        assert target_idx - LOOKBACK >= 0
        target_indices.append(target_idx)
    target_indices = np.asarray(target_indices, dtype=np.int64)

    if predictor["kind"] == "residual_mlp":
        x, _, baseline_scaled = build_residual_arrays(data, predictor["baseline"], target_indices, include_target=False)
        residual_scaled = predict_residual_mlp(model, x, device)
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

    model, history, residual_info = train_residual_mlp(data, best_baseline, device)
    print("Residual MLP validation:", {k: v for k, v in residual_info["metric"].items() if k != "horizon_rmse"})

    selected = choose_predictor(best_baseline, residual_info)
    final_metric = selected["metric"]
    print("selected predictor:", selected["name"])
    print("best_mse:", final_metric["mse"], "target_met:", final_metric["mse"] < MSE_TARGET)
    assert final_metric["mse"] <= float(baseline_df.loc[baseline_df["name"] == "last_value", "mse"].iloc[0])
    assert final_metric["mse"] < MSE_TARGET

    residual_backtest_df = rolling_residual_backtest(data, device) if residual_backtest else pd.DataFrame()

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
