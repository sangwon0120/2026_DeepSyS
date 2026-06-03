from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ett_forecasting_pytorch as ett


CSV_DIR = Path("csvFiles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and archive the existing SeqResidualBooster pipeline.")
    parser.add_argument("--experiment-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    parser.add_argument("--no-residual-backtest", action="store_true", help="Skip rolling residual retraining for a faster smoke run.")
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict:
    def run_git(*args: str) -> str:
        completed = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
        return completed.stdout.strip()

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(run_git("status", "--porcelain")),
        "dirty_files": run_git("status", "--porcelain").splitlines(),
    }


def save_table(table, path: Path) -> None:
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    frame.to_csv(path, index=False)
    print(f"saved: {path.resolve()} {frame.shape}")


def archive_files(experiment_id: str, paths: list[Path]) -> dict:
    archive_dir = CSV_DIR / "pytorch_experiments" / experiment_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for source in paths:
        archived = archive_dir / source.name
        shutil.copy2(source, archived)
        artifacts[source.name] = {
            "source": str(source.resolve()),
            "archived": str(archived),
            "sha256": sha256_file(source),
        }
    return artifacts


def main() -> None:
    args = parse_args()
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    submit_path = CSV_DIR / "submit_seq_residual.csv"
    report_path = CSV_DIR / "pytorch_run_report.json"
    ett.OUTPUT_PATH = str(submit_path)

    result = ett.run(plot=False, residual_backtest=not args.no_residual_backtest)
    t_cols = [f"T{i}" for i in range(ett.HORIZON)]
    table_paths = {
        "pytorch_training_history.csv": result["history"],
        "pytorch_refit_history.csv": result["refit_history"],
        "pytorch_baseline_candidates.csv": result["baseline"],
        "pytorch_baseline_backtest.csv": result["backtest"],
        "pytorch_residual_lambda_search.csv": result["residual_lambda"],
        "pytorch_residual_backtest.csv": result["residual_backtest"],
        "pytorch_submission_distribution.csv": result["submit"][t_cols].describe().loc[["mean", "std", "min", "max"]].reset_index(names="stat"),
    }
    for filename, table in table_paths.items():
        save_table(table, CSV_DIR / filename)

    archived_paths = [submit_path, *[CSV_DIR / filename for filename in table_paths]]
    artifacts = archive_files(args.experiment_id, archived_paths)
    report = {
        "experiment_id": args.experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git": git_metadata(),
        "settings": {
            "residual_backtest": not args.no_residual_backtest,
            "seq_error_lags": list(ett.SEQ_ERROR_LAGS),
            "pretrain_epochs": ett.PRETRAIN_EPOCHS,
            "finetune_epochs": ett.FINETUNE_EPOCHS,
            "residual_backtest_epochs": ett.RESIDUAL_BACKTEST_EPOCHS,
            "residual_lambdas": list(ett.RESIDUAL_LAMBDAS),
            "selected_training_plan": result["residual_training_plan"],
        },
        "selected_predictor": json_ready(result["predictor"]),
        "final_metric": json_ready(result["metric"]),
        "artifacts": artifacts,
        "kaggle_public_mse": None,
    }
    archive_report_path = CSV_DIR / "pytorch_experiments" / args.experiment_id / report_path.name
    report_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    archive_report_path.write_text(report_text, encoding="utf-8")
    print(f"saved: {report_path.resolve()}")
    print(f"archived: {archive_report_path.resolve()}")


if __name__ == "__main__":
    main()
