from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import ett_forecasting_pytorch as base


OUTPUT_PATH = "submit_ensemble.csv"
WEIGHTED_ENSEMBLE_NAME = "weighted_anchor_lgbm_catboost"
ENSEMBLE_LOOKBACK = 672
ENSEMBLE_START_LOOKBACK = ENSEMBLE_LOOKBACK + base.LOOKBACK
LAG_HOURS = (1, 2, 3, 6, 12, 24, 48, 72, 96, 168, 336, 672)
OT_ROLL_WINDOWS = (3, 6, 12, 24, 48, 96, 168, 336, 672)
RAW_ROLL_WINDOWS = (24, 168)
BASELINE_ERROR_LAGS = (96, 168, 336, 672)
WEIGHT_STEP = 0.05
ANCHOR_MIN_WEIGHT = 0.70
RESIDUAL_CLIP_SCALE = 2.0
RESIDUAL_LAMBDA_GRID = tuple(float(x) for x in np.round(np.arange(0.50, 1.601, 0.05), 2))
DEFAULT_CATBOOST_SEEDS = (base.SEED, 1337, 2026)


def default_num_threads() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def prepare_ensemble_data(
    df: pd.DataFrame,
    train_end_idx: int | None = None,
    train_targets: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
) -> dict:
    raw_df = base.clean_hourly_numeric_frame(df)
    test_start_idx = int(np.where(raw_df["date"].values == np.datetime64(base.TEST_START))[0][0])
    if train_end_idx is None:
        train_end_idx = int(test_start_idx * base.TRAIN_FRACTION_BEFORE_TEST)
        train_end_idx = (train_end_idx // 24) * 24

    if train_targets is None:
        train_targets = base.midnight_target_indices(raw_df, base.LOOKBACK, train_end_idx)
    model_train_targets = base.all_hour_target_indices(ENSEMBLE_START_LOOKBACK, train_end_idx)
    if val_targets is None:
        val_targets = base.midnight_target_indices(raw_df, train_end_idx, test_start_idx)

    clip_bounds = base.fit_feature_clip_bounds(raw_df, train_end_idx)
    clipped_df = base.apply_feature_clip(raw_df, clip_bounds)
    df_with_time = base.add_time_features(raw_df)
    feature_df = base.add_time_features(clipped_df)
    robust_center, robust_scale = fit_robust_scale(feature_df, train_end_idx)
    feature_df = add_robust_scaled_features(feature_df, robust_center, robust_scale)

    print("ensemble train target:", df_with_time.loc[train_targets[0], "date"], "~", df_with_time.loc[train_targets[-1] + base.HORIZON - 1, "date"])
    print("ensemble model target:", df_with_time.loc[model_train_targets[0], "date"], "~", df_with_time.loc[model_train_targets[-1] + base.HORIZON - 1, "date"])
    print("ensemble val target:", df_with_time.loc[val_targets[0], "date"], "~", df_with_time.loc[val_targets[-1] + base.HORIZON - 1, "date"])
    print("ensemble feature targets:", len(model_train_targets), "val targets:", len(val_targets))

    assert len(train_targets) > 0 and len(model_train_targets) > 0 and len(val_targets) > 0
    assert train_targets[0] - base.LOOKBACK >= 0
    assert model_train_targets[0] - ENSEMBLE_START_LOOKBACK >= 0
    assert val_targets[0] - ENSEMBLE_START_LOOKBACK >= 0
    assert train_targets[-1] + base.HORIZON <= train_end_idx
    assert model_train_targets[-1] + base.HORIZON <= train_end_idx
    assert val_targets[-1] + base.HORIZON <= test_start_idx
    assert (df_with_time.loc[train_targets, "date"].dt.hour == 0).all()
    assert (df_with_time.loc[val_targets, "date"].dt.hour == 0).all()

    return {
        "raw_df": raw_df,
        "df": df_with_time,
        "feature_df": feature_df,
        "clip_bounds": clip_bounds,
        "robust_center": robust_center,
        "robust_scale": robust_scale,
        "train_targets": train_targets,
        "model_train_targets": model_train_targets,
        "val_targets": val_targets,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
    }


def make_ensemble_data_view(data: dict, train_end_idx: int, train_targets: np.ndarray, val_targets: np.ndarray) -> dict:
    return prepare_ensemble_data(
        data["raw_df"],
        train_end_idx=train_end_idx,
        train_targets=train_targets,
        val_targets=val_targets,
    )


def fit_robust_scale(df: pd.DataFrame, train_end_idx: int) -> tuple[pd.Series, pd.Series]:
    train = df.loc[: train_end_idx - 1, base.RAW_COLS]
    center = train.median()
    q1 = train.quantile(0.25)
    q3 = train.quantile(0.75)
    scale = (q3 - q1).replace(0.0, np.nan)
    fallback = train.std().replace(0.0, 1.0)
    scale = scale.fillna(fallback).replace(0.0, 1.0)
    return center, scale


def add_robust_scaled_features(df: pd.DataFrame, center: pd.Series, scale: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for col in base.RAW_COLS:
        out[f"z_{col}"] = ((out[col] - float(center[col])) / float(scale[col])).astype("float32")
    return out


def build_tabular_residual_frame(
    data: dict,
    anchor_predictor: dict,
    target_indices: np.ndarray,
    include_target: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    df = data["df"]
    feature_df = data["feature_df"]
    raw_values = df[base.RAW_COLS].values.astype("float32")
    feature_values = feature_df[base.RAW_COLS].values.astype("float32")
    raw_ot = df[base.TARGET_COL].values.astype("float32")

    target_indices = np.asarray(target_indices, dtype=np.int64)
    assert np.all(target_indices - ENSEMBLE_START_LOOKBACK >= 0)
    assert np.all(target_indices + base.HORIZON <= len(df))

    anchor_pred = base.baseline_prediction(raw_ot, target_indices, anchor_predictor).astype("float32")
    last_value_pred = base.predict_last_value(raw_ot, target_indices).astype("float32")
    last_96h_pred = base.predict_last_96h_repeat(raw_ot, target_indices).astype("float32")
    last_week_pred = base.predict_last_week_same_time(raw_ot, target_indices).astype("float32")
    anchor_weights = anchor_weight_matrix(anchor_predictor)

    rows = []
    for row_idx, target_idx in enumerate(target_indices):
        origin_features, horizon_error_features = build_origin_features(
            df,
            feature_df,
            raw_values,
            feature_values,
            raw_ot,
            anchor_predictor,
            target_idx,
        )
        for horizon in range(base.HORIZON):
            target_row = int(target_idx + horizon)
            row = dict(origin_features)
            row.update(
                {
                    "horizon": float(horizon),
                    "horizon_day": float(horizon // 24),
                    "horizon_block": float(horizon // 24),
                    "horizon_sin": math.sin(2.0 * math.pi * horizon / base.HORIZON),
                    "horizon_cos": math.cos(2.0 * math.pi * horizon / base.HORIZON),
                    "target_hour": float(df.loc[target_row, "date"].hour),
                    "target_is_weekend": float(df.loc[target_row, "date"].dayofweek >= 5),
                    "anchor_pred": float(anchor_pred[row_idx, horizon]),
                    "last_value_pred": float(last_value_pred[row_idx, horizon]),
                    "last_96h_pred": float(last_96h_pred[row_idx, horizon]),
                    "last_week_pred": float(last_week_pred[row_idx, horizon]),
                    "anchor_minus_last_value": float(anchor_pred[row_idx, horizon] - last_value_pred[row_idx, horizon]),
                    "anchor_minus_last_96h": float(anchor_pred[row_idx, horizon] - last_96h_pred[row_idx, horizon]),
                    "anchor_minus_last_week": float(anchor_pred[row_idx, horizon] - last_week_pred[row_idx, horizon]),
                    "last_96h_minus_week": float(last_96h_pred[row_idx, horizon] - last_week_pred[row_idx, horizon]),
                    "w_last": float(anchor_weights[0, horizon]),
                    "w_last_96h": float(anchor_weights[1, horizon]),
                    "w_week": float(anchor_weights[2, horizon]),
                }
            )
            for name, values in horizon_error_features.items():
                row[name] = float(values[horizon])
            for col in base.TIME_COLS:
                row[f"target_{col}"] = float(df.loc[target_row, col])
            rows.append(row)

    x = pd.DataFrame(rows)
    y = None
    if include_target:
        true = base.make_true(raw_ot, target_indices).astype("float32")
        y = (true - anchor_pred).reshape(-1).astype("float32")
    return clean_feature_frame(x), y, anchor_pred


def build_origin_features(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    raw_values: np.ndarray,
    feature_values: np.ndarray,
    raw_ot: np.ndarray,
    anchor_predictor: dict,
    target_idx: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    date = df.loc[target_idx, "date"]
    row: dict[str, float] = {
        "origin_hour": float(date.hour),
        "origin_dow": float(date.dayofweek),
        "origin_month": float(date.month),
        "origin_is_weekend": float(date.dayofweek >= 5),
        "origin_day_index": float(target_idx // 24),
    }
    for col in base.TIME_COLS:
        row[f"origin_{col}"] = float(df.loc[target_idx, col])

    ot_idx = base.RAW_COLS.index(base.TARGET_COL)
    ot = feature_values[:, ot_idx]
    for lag in LAG_HOURS:
        row[f"ot_lag_{lag}"] = float(ot[target_idx - lag])
    for col_idx, col in enumerate(base.RAW_COLS):
        row[f"{col}_last"] = float(feature_values[target_idx - 1, col_idx])
        row[f"{col}_lag_24"] = float(feature_values[target_idx - 24, col_idx])
        row[f"{col}_lag_168"] = float(feature_values[target_idx - 168, col_idx])
        row[f"z_{col}_last"] = float(feature_df.loc[target_idx - 1, f"z_{col}"])

    for window in OT_ROLL_WINDOWS:
        values = ot[target_idx - window : target_idx]
        row[f"ot_mean_{window}"] = float(values.mean())
        row[f"ot_std_{window}"] = float(values.std())
        row[f"ot_min_{window}"] = float(values.min())
        row[f"ot_max_{window}"] = float(values.max())
        row[f"ot_last_minus_mean_{window}"] = float(ot[target_idx - 1] - values.mean())

    for window in RAW_ROLL_WINDOWS:
        values = feature_values[target_idx - window : target_idx]
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        for col_idx, col in enumerate(base.RAW_COLS):
            row[f"{col}_mean_{window}"] = float(means[col_idx])
            row[f"{col}_std_{window}"] = float(stds[col_idx])

    row["ot_delta_24"] = float(ot[target_idx - 1] - ot[target_idx - 25])
    row["ot_delta_168"] = float(ot[target_idx - 1] - ot[target_idx - 169])
    row["ot_mean_24_minus_168"] = float(row["ot_mean_24"] - row["ot_mean_168"])
    row["ot_mean_168_minus_672"] = float(row["ot_mean_168"] - row["ot_mean_672"])
    row["raw_ot_last"] = float(raw_values[target_idx - 1, ot_idx])
    error_summary, horizon_error_features = build_past_anchor_error_features(raw_ot, anchor_predictor, target_idx)
    row.update(error_summary)
    return row, horizon_error_features


def build_past_anchor_error_features(
    raw_ot: np.ndarray,
    anchor_predictor: dict,
    target_idx: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    summary: dict[str, float] = {}
    by_horizon: dict[str, np.ndarray] = {}
    past_starts = np.asarray([target_idx - lag for lag in BASELINE_ERROR_LAGS], dtype=np.int64)
    assert np.all(past_starts - base.LOOKBACK >= 0)
    assert np.all(past_starts + base.HORIZON <= target_idx)

    past_pred = base.baseline_prediction(raw_ot, past_starts, anchor_predictor).astype("float32")
    for lag_idx, lag in enumerate(BASELINE_ERROR_LAGS):
        past_start = int(past_starts[lag_idx])
        err = (raw_ot[past_start : past_start + base.HORIZON] - past_pred[lag_idx]).astype("float32")
        abs_err = np.abs(err)
        summary[f"anchor_err_lag_{lag}_mean"] = float(err.mean())
        summary[f"anchor_err_lag_{lag}_std"] = float(err.std())
        summary[f"anchor_err_lag_{lag}_abs_mean"] = float(abs_err.mean())
        summary[f"anchor_err_lag_{lag}_last"] = float(err[-1])
        summary[f"anchor_err_lag_{lag}_sign_mean"] = float(np.sign(err).mean())
        for block_start in range(0, base.HORIZON, 24):
            block = err[block_start : block_start + 24]
            summary[f"anchor_err_lag_{lag}_block{block_start // 24}_mean"] = float(block.mean())
        by_horizon[f"anchor_err_lag_{lag}_same_h"] = err
        by_horizon[f"anchor_abs_err_lag_{lag}_same_h"] = abs_err
    return summary, by_horizon


def anchor_weight_matrix(anchor_predictor: dict) -> np.ndarray:
    if anchor_predictor["name"].startswith("threeway_"):
        return np.asarray(anchor_predictor["weights"], dtype=np.float32)
    weights = np.zeros((3, base.HORIZON), dtype=np.float32)
    if anchor_predictor["name"] == "last_value":
        weights[0, :] = 1.0
    elif anchor_predictor["name"] == "last_96h_repeat":
        weights[1, :] = 1.0
    elif anchor_predictor["name"] == "last_week_same_time":
        weights[2, :] = 1.0
    else:
        weights[0, :] = 1.0
    return weights


def clean_feature_frame(x: pd.DataFrame) -> pd.DataFrame:
    out = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype("float32")


def fit_horizon_residual_norm(y_flat: np.ndarray, n_targets: int) -> tuple[np.ndarray, np.ndarray]:
    residual = y_flat.reshape(n_targets, base.HORIZON)
    center = residual.mean(axis=0).astype("float32")
    scale = residual.std(axis=0).astype("float32")
    scale = np.where(scale < 1e-3, 1.0, scale).astype("float32")
    return center, scale


def normalize_residual(y_flat: np.ndarray, n_targets: int, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    residual = y_flat.reshape(n_targets, base.HORIZON)
    return ((residual - center[None, :]) / scale[None, :]).reshape(-1).astype("float32")


def denormalize_residual_matrix(residual: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (residual * scale[None, :] + center[None, :]).astype("float32")


def postprocess_residuals(residuals: list[np.ndarray], horizon_scale: np.ndarray) -> list[np.ndarray]:
    if not residuals:
        return []

    clip = RESIDUAL_CLIP_SCALE * horizon_scale[None, :]
    clipped = [np.clip(residual, -clip, clip).astype("float32") for residual in residuals]
    if len(clipped) < 2:
        return clipped

    signs = [np.sign(residual) for residual in clipped]
    agree = np.ones_like(clipped[0], dtype=bool)
    for sign in signs[1:]:
        agree &= signs[0] * sign > 0
    return [np.where(agree, residual, 0.0).astype("float32") for residual in clipped]


def clip_residual(residual: np.ndarray, horizon_scale: np.ndarray) -> np.ndarray:
    clip = RESIDUAL_CLIP_SCALE * horizon_scale[None, :]
    return np.clip(residual, -clip, clip).astype("float32")


def select_residual_lambda(
    anchor_pred: np.ndarray,
    residual: np.ndarray,
    true: np.ndarray,
    lambdas: tuple[float, ...] = RESIDUAL_LAMBDA_GRID,
) -> tuple[float, np.ndarray, dict, pd.DataFrame]:
    rows = []
    best_lambda = 0.0
    best_pred = anchor_pred
    best_metric = base.score_raw(anchor_pred, true)
    best_mse = float(best_metric["mse"])

    for value in lambdas:
        pred = anchor_pred + float(value) * residual
        score = base.score_raw(pred, true)
        rows.append({"lambda": float(value), **metric_row(score)})
        if score["mse"] < best_mse:
            best_lambda = float(value)
            best_pred = pred
            best_metric = score
            best_mse = float(score["mse"])

    return best_lambda, best_pred.astype("float32"), best_metric, pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def safe_col_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def fit_lightgbm_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    num_threads: int | None = None,
):
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed; skipping lightgbm_residual.")
        return None

    threads = int(num_threads or default_num_threads())
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="l2",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.20,
        random_state=base.SEED,
        n_jobs=threads,
        force_col_wise=True,
        verbosity=-1,
    )
    try:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
    except TypeError:
        model.fit(x_train, y_train)
    return model


def fit_catboost_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    num_threads: int | None = None,
    use_gpu: bool = False,
    seed: int = base.SEED,
):
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("CatBoost not installed; skipping catboost_residual.")
        return None

    threads = int(num_threads or default_num_threads())
    params = dict(
        loss_function="RMSE",
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=int(seed),
        allow_writing_files=False,
        verbose=False,
    )
    if use_gpu:
        params.update({"task_type": "GPU", "devices": "0"})
    else:
        params["thread_count"] = threads

    model = CatBoostRegressor(**params)
    try:
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, verbose=False)
    except Exception as exc:
        if not use_gpu:
            raise
        print(f"CatBoost GPU failed; falling back to CPU. Reason: {exc}")
        params.pop("task_type", None)
        params.pop("devices", None)
        params["thread_count"] = threads
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, verbose=False)
    return model


def fit_catboost_models(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    num_threads: int | None = None,
    use_gpu: bool = False,
    seeds: tuple[int, ...] = DEFAULT_CATBOOST_SEEDS,
) -> list:
    models = []
    for seed in seeds:
        model = fit_catboost_model(
            x_train,
            y_train,
            x_val,
            y_val,
            num_threads=num_threads,
            use_gpu=use_gpu,
            seed=int(seed),
        )
        if model is not None:
            models.append(model)
    return models


def train_ensemble_split(
    data: dict,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    use_lightgbm: bool = True,
    use_catboost: bool = True,
    num_threads: int | None = None,
    catboost_gpu: bool = False,
    catboost_seeds: tuple[int, ...] = DEFAULT_CATBOOST_SEEDS,
    label: str = "Main",
) -> dict:
    train_targets = np.asarray(train_targets, dtype=np.int64)

    split_data = make_ensemble_data_view(data, data["train_end_idx"], train_targets, val_targets)
    model_train_targets = split_data["model_train_targets"]
    assert len(model_train_targets) > 0
    baseline_df, anchor = base.fit_baseline_candidates(split_data)
    print(f"{label} anchor:", anchor["name"], anchor["metric"]["mse"])

    x_train, y_train_raw, _ = build_tabular_residual_frame(split_data, anchor, model_train_targets)
    x_val, y_val_raw, anchor_val = build_tabular_residual_frame(split_data, anchor, val_targets)
    feature_cols = list(x_train.columns)
    x_val = x_val.reindex(columns=feature_cols, fill_value=0.0).astype("float32")
    residual_center, residual_scale = fit_horizon_residual_norm(y_train_raw, len(model_train_targets))
    y_train = normalize_residual(y_train_raw, len(model_train_targets), residual_center, residual_scale)
    y_val = normalize_residual(y_val_raw, len(val_targets), residual_center, residual_scale)

    raw_ot = split_data["df"][base.TARGET_COL].values.astype("float32")
    val_true = base.make_true(raw_ot, val_targets).astype("float32")

    candidate_names = ["anchor"]
    candidate_preds = [anchor_val]
    candidate_metrics = {"anchor": base.score_raw(anchor_val, val_true)}
    val_candidate_preds = {"anchor": anchor_val}
    models = {}
    candidate_specs = {"anchor": {"kind": "anchor", "lambda": 0.0}}
    lambda_frames = []
    rows = [{"name": "anchor", **metric_row(candidate_metrics["anchor"])}]

    if use_lightgbm:
        model = fit_lightgbm_model(x_train, y_train, x_val, y_val, num_threads=num_threads)
        if model is not None:
            residual_norm = model.predict(x_val).reshape(len(val_targets), base.HORIZON).astype("float32")
            residual = clip_residual(denormalize_residual_matrix(residual_norm, residual_center, residual_scale), residual_scale)
            best_lambda, pred, pred_metric, lambda_df = select_residual_lambda(anchor_val, residual, val_true)
            models["lightgbm_residual"] = model
            candidate_names.append("lightgbm_residual")
            candidate_preds.append(pred)
            candidate_metrics["lightgbm_residual"] = pred_metric
            val_candidate_preds["lightgbm_residual"] = pred
            candidate_specs["lightgbm_residual"] = {"kind": "residual", "model": "lightgbm_residual", "lambda": best_lambda}
            lambda_df.insert(0, "candidate", "lightgbm_residual")
            lambda_frames.append(lambda_df)
            rows.append({"name": "lightgbm_residual", "lambda": best_lambda, **metric_row(pred_metric)})

    if use_catboost:
        catboost_models = fit_catboost_models(
            x_train,
            y_train,
            x_val,
            y_val,
            num_threads=num_threads,
            use_gpu=catboost_gpu,
            seeds=catboost_seeds,
        )
        if catboost_models:
            residual_norms = [model.predict(x_val).reshape(len(val_targets), base.HORIZON).astype("float32") for model in catboost_models]
            residual_norm = np.mean(residual_norms, axis=0).astype("float32")
            residual = clip_residual(denormalize_residual_matrix(residual_norm, residual_center, residual_scale), residual_scale)
            best_lambda, pred, pred_metric, lambda_df = select_residual_lambda(anchor_val, residual, val_true)
            models["catboost_residual"] = catboost_models
            candidate_names.append("catboost_residual")
            candidate_preds.append(pred)
            candidate_metrics["catboost_residual"] = pred_metric
            val_candidate_preds["catboost_residual"] = pred
            candidate_specs["catboost_residual"] = {
                "kind": "residual",
                "model": "catboost_residual",
                "lambda": best_lambda,
                "seeds": tuple(int(seed) for seed in catboost_seeds),
            }
            lambda_df.insert(0, "candidate", "catboost_residual")
            lambda_frames.append(lambda_df)
            rows.append({"name": "catboost_residual", "lambda": best_lambda, **metric_row(pred_metric)})

    weights = fit_block_ensemble_weights(val_true, candidate_preds)
    ensemble_val = apply_block_weights(candidate_preds, weights)
    ensemble_metric = base.score_raw(ensemble_val, val_true)
    candidate_metrics[WEIGHTED_ENSEMBLE_NAME] = ensemble_metric
    val_candidate_preds[WEIGHTED_ENSEMBLE_NAME] = ensemble_val
    candidate_specs[WEIGHTED_ENSEMBLE_NAME] = {"kind": "weighted", "lambda": np.nan}
    rows.append({"name": WEIGHTED_ENSEMBLE_NAME, "lambda": np.nan, **metric_row(ensemble_metric)})
    candidate_df = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    if "lambda" not in candidate_df.columns:
        candidate_df["lambda"] = np.nan
    weight_df = block_weight_frame(weights, candidate_names)
    lambda_df = pd.concat(lambda_frames, ignore_index=True) if lambda_frames else pd.DataFrame()

    print(f"{label} candidates")
    print(candidate_df)
    print(f"{label} ensemble metric:", metric_row(ensemble_metric))
    print(f"{label} ensemble weights")
    print(weight_df)

    return {
        "data": split_data,
        "baseline": baseline_df,
        "anchor": anchor,
        "models": models,
        "feature_cols": feature_cols,
        "candidate_names": candidate_names,
        "candidate_df": candidate_df,
        "candidate_metrics": candidate_metrics,
        "val_candidate_preds": val_candidate_preds,
        "candidate_specs": candidate_specs,
        "lambda_df": lambda_df,
        "weights": weights,
        "weights_df": weight_df,
        "residual_center": residual_center,
        "residual_scale": residual_scale,
        "metric": ensemble_metric,
        "anchor_metric": base.score_raw(anchor_val, val_true),
        "val_pred": ensemble_val,
        "val_true": val_true,
    }


def metric_row(metric: dict) -> dict:
    return {
        "mse": float(metric["mse"]),
        "rmse": float(metric["rmse"]),
        "mae": float(metric["mae"]),
    }


def fit_block_ensemble_weights(true: np.ndarray, preds: list[np.ndarray]) -> np.ndarray:
    n_models = len(preds)
    if n_models == 1:
        return np.ones((4, 1), dtype=np.float32)

    weights = np.zeros((4, n_models), dtype=np.float32)
    for block_idx, start in enumerate(range(0, base.HORIZON, 24)):
        end = start + 24
        weights[block_idx] = fit_weights(true[:, start:end], [pred[:, start:end] for pred in preds])
    return weights


def fit_weights(true: np.ndarray, preds: list[np.ndarray]) -> np.ndarray:
    n_models = len(preds)
    if n_models == 1:
        return np.ones(1, dtype=np.float32)
    if n_models <= 4:
        return fit_weights_grid(true, preds)
    return fit_weights_coordinate(true, preds)


def fit_weights_grid(true: np.ndarray, preds: list[np.ndarray]) -> np.ndarray:
    n_models = len(preds)
    units = int(round(1.0 / WEIGHT_STEP))
    min_anchor_units = int(math.ceil(ANCHOR_MIN_WEIGHT / WEIGHT_STEP))
    best_loss = float("inf")
    best_units = [units] + [0] * (n_models - 1)

    def search(index: int, remaining: int, current: list[int]) -> None:
        nonlocal best_loss, best_units
        if index == n_models - 1:
            candidate = current + [remaining]
            if candidate[0] < min_anchor_units:
                return
            w = np.asarray(candidate, dtype=np.float32) / units
            pred = sum(float(w[i]) * preds[i] for i in range(n_models))
            loss = float(np.mean((pred - true) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_units = candidate
            return

        for value in range(remaining + 1):
            if index == 0 and value < min_anchor_units:
                continue
            search(index + 1, remaining - value, current + [value])

    search(0, units, [])
    return (np.asarray(best_units, dtype=np.float32) / units).astype("float32")


def fit_weights_coordinate(true: np.ndarray, preds: list[np.ndarray]) -> np.ndarray:
    n_models = len(preds)
    weights = np.zeros(n_models, dtype=np.float32)
    weights[0] = 1.0
    step = np.float32(WEIGHT_STEP)

    def loss(w: np.ndarray) -> float:
        pred = sum(float(w[i]) * preds[i] for i in range(n_models))
        return float(np.mean((pred - true) ** 2))

    best_loss = loss(weights)
    improved = True
    while improved:
        improved = False
        for src in range(n_models):
            for dst in range(n_models):
                if src == dst or weights[src] < step:
                    continue
                trial = weights.copy()
                trial[src] -= step
                trial[dst] += step
                if trial[0] < ANCHOR_MIN_WEIGHT:
                    continue
                trial_loss = loss(trial)
                if trial_loss + 1e-9 < best_loss:
                    weights = trial
                    best_loss = trial_loss
                    improved = True
    return weights


def apply_block_weights(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(preds[0], dtype=np.float32)
    for block_idx, start in enumerate(range(0, base.HORIZON, 24)):
        end = start + 24
        for model_idx, pred in enumerate(preds):
            out[:, start:end] += float(weights[block_idx, model_idx]) * pred[:, start:end]
    return out


def block_weight_frame(weights: np.ndarray, candidate_names: list[str]) -> pd.DataFrame:
    rows = []
    for block_idx, start in enumerate(range(0, base.HORIZON, 24)):
        row = {"block": f"T{start}-T{start + 23}"}
        for model_idx, name in enumerate(candidate_names):
            row[name] = float(weights[block_idx, model_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def predict_candidate_targets_map(artifacts: dict, data: dict, target_indices: np.ndarray) -> dict[str, np.ndarray]:
    x, _, anchor_pred = build_tabular_residual_frame(data, artifacts["anchor"], target_indices, include_target=False)
    x = x.reindex(columns=artifacts["feature_cols"], fill_value=0.0).astype("float32")

    pred_map = {"anchor": anchor_pred}
    residual_names = artifacts["candidate_names"][1:]
    for name in residual_names:
        spec = artifacts["candidate_specs"][name]
        model = artifacts["models"][name]
        if isinstance(model, list):
            residual_norms = [m.predict(x).reshape(len(target_indices), base.HORIZON).astype("float32") for m in model]
            residual_norm = np.mean(residual_norms, axis=0).astype("float32")
        else:
            residual_norm = model.predict(x).reshape(len(target_indices), base.HORIZON).astype("float32")
        residual = denormalize_residual_matrix(residual_norm, artifacts["residual_center"], artifacts["residual_scale"])
        residual = clip_residual(residual, artifacts["residual_scale"])
        pred_map[name] = anchor_pred + float(spec["lambda"]) * residual

    weighted_inputs = [pred_map[name] for name in artifacts["candidate_names"]]
    pred_map[WEIGHTED_ENSEMBLE_NAME] = apply_block_weights(weighted_inputs, artifacts["weights"])
    return pred_map


def predict_candidate_targets(artifacts: dict, data: dict, target_indices: np.ndarray, candidate_name: str) -> np.ndarray:
    pred_map = predict_candidate_targets_map(artifacts, data, target_indices)
    if candidate_name not in pred_map:
        raise ValueError(f"unknown prediction candidate: {candidate_name}")
    return pred_map[candidate_name]


def predict_ensemble_targets(artifacts: dict, data: dict, target_indices: np.ndarray) -> np.ndarray:
    return predict_candidate_targets(artifacts, data, target_indices, WEIGHTED_ENSEMBLE_NAME)


def rolling_ensemble_backtest(
    data: dict,
    use_lightgbm: bool = True,
    use_catboost: bool = True,
    num_threads: int | None = None,
    catboost_gpu: bool = False,
    catboost_seeds: tuple[int, ...] = DEFAULT_CATBOOST_SEEDS,
) -> pd.DataFrame:
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

        inner_train_end = int(outer_start * base.TRAIN_FRACTION_BEFORE_TEST)
        inner_train_end = (inner_train_end // 24) * 24
        inner_train_targets = base.midnight_target_indices(df, base.LOOKBACK, inner_train_end)
        inner_val_targets = base.midnight_target_indices(df, inner_train_end, outer_start)
        outer_targets = base.midnight_target_indices(df, outer_start, outer_end)

        if len(inner_train_targets) == 0 or len(inner_val_targets) == 0 or len(outer_targets) == 0:
            continue
        if inner_train_targets[-1] < ENSEMBLE_START_LOOKBACK or outer_targets[0] < ENSEMBLE_START_LOOKBACK:
            continue

        inner_data = make_ensemble_data_view(data, inner_train_end, inner_train_targets, inner_val_targets)
        result = train_ensemble_split(
            inner_data,
            inner_data["train_targets"],
            inner_data["val_targets"],
            use_lightgbm=use_lightgbm,
            use_catboost=use_catboost,
            num_threads=num_threads,
            catboost_gpu=catboost_gpu,
            catboost_seeds=catboost_seeds,
            label=f"EnsBT{fold_idx}",
        )

        raw_ot = inner_data["df"][base.TARGET_COL].values.astype("float32")
        outer_true = base.make_true(raw_ot, outer_targets).astype("float32")
        baseline_outer_pred = base.baseline_prediction(raw_ot, outer_targets, result["anchor"]).astype("float32")
        selected_candidate = str(result["candidate_df"].iloc[0]["name"])
        outer_pred_map = predict_candidate_targets_map(result, inner_data, outer_targets)
        selected_outer_pred = outer_pred_map[selected_candidate]

        baseline_outer = base.score_raw(baseline_outer_pred, outer_true)
        outer_scores = {name: base.score_raw(pred, outer_true) for name, pred in outer_pred_map.items()}
        selected_outer = base.score_raw(selected_outer_pred, outer_true)
        row = {
            "fold": fold_idx,
            "outer_start": df.loc[outer_targets[0], "date"],
            "outer_end": df.loc[outer_targets[-1] + base.HORIZON - 1, "date"],
            "baseline": result["anchor"]["name"],
            "inner_baseline_mse": result["anchor_metric"]["mse"],
            "inner_ensemble_mse": result["metric"]["mse"],
            "inner_selected_candidate": selected_candidate,
            "inner_selected_mse": result["candidate_metrics"][selected_candidate]["mse"],
            "outer_baseline_mse": baseline_outer["mse"],
            "outer_ensemble_mse": outer_scores[WEIGHTED_ENSEMBLE_NAME]["mse"],
            "outer_selected_mse": selected_outer["mse"],
            "outer_selected_mae": selected_outer["mae"],
            "improvement": baseline_outer["mse"] - selected_outer["mse"],
        }
        for name, score in outer_scores.items():
            row[f"outer_{safe_col_name(name)}_mse"] = score["mse"]
            row[f"outer_{safe_col_name(name)}_mae"] = score["mae"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    summary = {
        "folds": len(out),
        "baseline_mean_mse": out["outer_baseline_mse"].mean(),
        "ensemble_mean_mse": out["outer_ensemble_mse"].mean(),
        "selected_mean_mse": out["outer_selected_mse"].mean(),
        "baseline_max_mse": out["outer_baseline_mse"].max(),
        "ensemble_max_mse": out["outer_ensemble_mse"].max(),
        "selected_max_mse": out["outer_selected_mse"].max(),
        "mean_improvement": out["improvement"].mean(),
        "positive_folds": int((out["improvement"] > 0).sum()),
    }
    print("rolling ensemble backtest")
    print(out)
    print("rolling ensemble summary")
    print(pd.DataFrame([summary]))
    return out


def ensemble_backtest_passes(backtest: pd.DataFrame) -> tuple[bool, str]:
    if backtest is None or backtest.empty:
        return False, "ensemble backtest was not run"

    mean_improvement = float(backtest["improvement"].mean())
    positive_folds = int((backtest["improvement"] > 0).sum())
    folds = int(len(backtest))
    selected_max = float(backtest["outer_selected_mse"].max())
    baseline_max = float(backtest["outer_baseline_mse"].max())

    passes = mean_improvement > 0.0 and positive_folds >= max(1, (folds // 2) + 1) and selected_max <= baseline_max * 1.05
    reason = (
        f"ensemble backtest mean_improvement={mean_improvement:.6f}, "
        f"positive_folds={positive_folds}/{folds}, "
        f"selected_max={selected_max:.6f}, baseline_max={baseline_max:.6f}"
    )
    return passes, reason


def choose_ensemble_predictor(result: dict, backtest: pd.DataFrame | None) -> dict:
    candidate_df = result["candidate_df"]
    best_candidate = str(candidate_df.iloc[0]["name"])
    best_metric = result["candidate_metrics"][best_candidate]
    best_spec = result["candidate_specs"][best_candidate]

    if best_candidate == "anchor":
        return {
            "name": result["anchor"]["name"],
            "kind": "baseline",
            "candidate": "anchor",
            "lambda": 0.0,
            "metric": best_metric,
            "selection_reason": "baseline selected because it has the lowest validation MSE",
        }

    kind = "ensemble" if best_candidate == WEIGHTED_ENSEMBLE_NAME else "residual_model"
    name = best_candidate

    if backtest is not None:
        _, reason = ensemble_backtest_passes(backtest)
        selection_reason = f"{best_candidate} selected because it has the lowest validation MSE; rolling backtest diagnostic: {reason}"
    else:
        selection_reason = f"{best_candidate} selected because it has the lowest validation MSE; rolling gate not run"

    return {
        "name": name,
        "kind": kind,
        "candidate": best_candidate,
        "lambda": best_spec.get("lambda", np.nan),
        "metric": best_metric,
        "selection_reason": selection_reason,
    }


def submission_target_indices(df: pd.DataFrame, sub: pd.DataFrame, id_col: str, selected: dict) -> np.ndarray:
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(df["date"])}
    indices = []
    for ts in sub[id_col]:
        target_ts = pd.Timestamp(ts)
        target_idx = date_to_idx[target_ts]
        assert df.loc[target_idx - 1, "date"] < target_ts
        if selected["kind"] in ("ensemble", "residual_model"):
            assert target_idx - ENSEMBLE_START_LOOKBACK >= 0
        else:
            assert target_idx - base.LOOKBACK >= 0
        indices.append(target_idx)
    return np.asarray(indices, dtype=np.int64)


def make_submission(
    selected: dict,
    result: dict,
    data: dict,
    sub: pd.DataFrame,
    id_col: str,
    output_path: str | Path = OUTPUT_PATH,
) -> pd.DataFrame:
    target_indices = submission_target_indices(data["df"], sub, id_col, selected)
    raw_ot = data["df"][base.TARGET_COL].values.astype("float32")

    if selected["kind"] in ("ensemble", "residual_model"):
        pred_values = predict_candidate_targets(result, data, target_indices, selected["candidate"])
    else:
        pred_values = base.baseline_prediction(raw_ot, target_indices, result["anchor"]).astype("float32")

    t_cols = [f"T{i}" for i in range(base.HORIZON)]
    out = sub.copy()
    out[t_cols] = pred_values
    out[id_col] = pd.to_datetime(out[id_col]).dt.strftime("%Y-%m-%d")
    out = out[[id_col] + t_cols]
    out.to_csv(output_path, index=False)

    print("selected predictor:", selected["name"])
    print("selection reason:", selected["selection_reason"])
    print("saved:", output_path, out.shape)
    print(out.head())
    print(out.tail())
    print(out[t_cols].describe().loc[["mean", "std", "min", "max"]])

    assert out.shape == sub[[id_col] + t_cols].shape
    assert np.isfinite(out[t_cols].values).all()
    return out


def format_submission_frame(sub: pd.DataFrame, id_col: str, pred_values: np.ndarray) -> pd.DataFrame:
    t_cols = [f"T{i}" for i in range(base.HORIZON)]
    out = sub.copy()
    out[t_cols] = pred_values
    out[id_col] = pd.to_datetime(out[id_col]).dt.strftime("%Y-%m-%d")
    out = out[[id_col] + t_cols]
    assert out.shape == sub[[id_col] + t_cols].shape
    assert np.isfinite(out[t_cols].values).all()
    return out


def write_candidate_submissions(
    result: dict,
    data: dict,
    sub: pd.DataFrame,
    id_col: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    target_indices = submission_target_indices(
        data["df"],
        sub,
        id_col,
        {"kind": "residual_model", "candidate": "catboost_residual"},
    )
    pred_map = predict_candidate_targets_map(result, data, target_indices)

    paths: dict[str, Path] = {}
    for name in result["candidate_df"]["name"].tolist():
        candidate = "anchor" if name == "anchor" else str(name)
        if candidate not in pred_map:
            continue
        path = output_dir / f"submit_{safe_col_name(candidate)}.csv"
        format_submission_frame(sub, id_col, pred_map[candidate]).to_csv(path, index=False)
        paths[candidate] = path

    print("\ncandidate submissions")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return paths


def horizon_summary(metric: dict) -> pd.DataFrame:
    rmse = np.asarray(metric["horizon_rmse"], dtype=np.float32)
    rows = [
        {"segment": "all", "rmse_mean": float(rmse.mean()), "rmse_max": float(rmse.max()), "argmax": int(rmse.argmax())},
    ]
    for start in range(0, base.HORIZON, 24):
        block = rmse[start : start + 24]
        rows.append(
            {
                "segment": f"T{start}-T{start + 23}",
                "rmse_mean": float(block.mean()),
                "rmse_max": float(block.max()),
                "argmax": int(start + block.argmax()),
            }
        )
    return pd.DataFrame(rows)


def submission_distribution(submit: pd.DataFrame) -> pd.DataFrame:
    t_cols = [f"T{i}" for i in range(base.HORIZON)]
    return submit[t_cols].describe().loc[["mean", "std", "min", "max"]]


def print_run_diagnostics(result: dict, selected: dict, submit: pd.DataFrame, backtest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    h_summary = horizon_summary(selected["metric"])
    s_dist = submission_distribution(submit)

    print("\nvalidation candidates")
    print(result["candidate_df"])
    if not result["lambda_df"].empty:
        print("\nresidual lambda search")
        print(result["lambda_df"].groupby("candidate", as_index=False).head(5).reset_index(drop=True))
    print("\nensemble weights")
    print(result["weights_df"])
    if backtest is not None and not backtest.empty:
        print("\nrolling backtest")
        print(backtest)
    print("\nhorizon RMSE summary")
    print(h_summary)
    print("\nsubmission distribution")
    print(s_dist)
    return h_summary, s_dist


def plot_result(result: dict, selected: dict) -> None:
    import matplotlib.pyplot as plt

    val_true = result["val_true"]
    val_pred = result["val_candidate_preds"][selected["candidate"]]
    rmse = np.sqrt(np.mean((val_pred - val_true) ** 2, axis=0))

    plt.figure(figsize=(9, 4))
    plt.plot(rmse)
    plt.title("Selected validation RMSE by horizon")
    plt.xlabel("horizon")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()


def run_ensemble(
    plot: bool = True,
    full_backtest: bool = True,
    use_lightgbm: bool = True,
    use_catboost: bool = True,
    num_threads: int | None = None,
    catboost_gpu: bool = False,
    catboost_seeds: tuple[int, ...] = DEFAULT_CATBOOST_SEEDS,
    output_path: str | Path = OUTPUT_PATH,
) -> dict:
    base.set_seed()
    threads = int(num_threads or default_num_threads())
    print("local profile:", {"threads": threads, "catboost_gpu": bool(catboost_gpu), "catboost_seeds": tuple(catboost_seeds)})
    df, sub, id_col = base.load_patchtst_data()
    data = prepare_ensemble_data(df)

    result = train_ensemble_split(
        data,
        data["train_targets"],
        data["val_targets"],
        use_lightgbm=use_lightgbm,
        use_catboost=use_catboost,
        num_threads=threads,
        catboost_gpu=catboost_gpu,
        catboost_seeds=catboost_seeds,
    )
    backtest = (
        rolling_ensemble_backtest(
            data,
            use_lightgbm=use_lightgbm,
            use_catboost=use_catboost,
            num_threads=threads,
            catboost_gpu=catboost_gpu,
            catboost_seeds=catboost_seeds,
        )
        if full_backtest
        else pd.DataFrame()
    )
    selected = choose_ensemble_predictor(result, backtest if full_backtest else None)
    final_metric = selected["metric"]

    print("final selected:", selected["name"])
    print("final metric:", metric_row(final_metric))

    if plot:
        plot_result(result, selected)

    submit = make_submission(selected, result, data, sub, id_col, output_path)
    candidate_submission_paths = write_candidate_submissions(result, data, sub, id_col, Path(output_path).parent)
    h_summary, s_dist = print_run_diagnostics(result, selected, submit, backtest)
    return {
        "baseline": result["baseline"],
        "candidate": result["candidate_df"],
        "weights": result["weights_df"],
        "lambda": result["lambda_df"],
        "horizon_summary": h_summary,
        "submission_distribution": s_dist,
        "metric": final_metric,
        "predictor": selected,
        "backtest": backtest,
        "submit": submit,
        "candidate_submissions": candidate_submission_paths,
        "artifacts": result,
    }


if __name__ == "__main__":
    run_ensemble(plot=True, full_backtest=True)
