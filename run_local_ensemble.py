from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ETTh1 ensemble pipeline locally.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("csvFiles/submit_ensemble.csv"))
    parser.add_argument("--plot", action="store_true", help="Show validation RMSE plot.")
    parser.add_argument("--quick", action="store_true", help="Skip rolling backtest and CatBoost for a fast smoke test.")
    parser.add_argument("--no-backtest", action="store_true", help="Skip rolling backtest.")
    parser.add_argument("--no-lightgbm", action="store_true", help="Disable LightGBM residual model.")
    parser.add_argument("--no-catboost", action="store_true", help="Disable CatBoost residual model.")
    parser.add_argument("--threads", type=int, default=None, help="CPU worker threads. Default keeps two logical threads free.")
    parser.add_argument("--catboost-gpu", action="store_true", help="Try CatBoost on RTX GPU, then fall back to CPU if unavailable.")
    parser.add_argument("--catboost-seeds", default="42,1337,2026", help="Comma-separated CatBoost seeds for seed ensembling.")
    parser.add_argument("--experiment-id", default=None, help="Archive ID. Default is the current UTC timestamp.")
    return parser.parse_args()


def default_threads() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("--catboost-seeds must contain at least one integer seed.")
    return seeds


def default_experiment_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def validate_experiment_id(value: str) -> str:
    if not value or any(not (ch.isalnum() or ch in "._-") for ch in value):
        raise ValueError("--experiment-id must contain only letters, numbers, '.', '_', or '-'.")
    return value


def git_metadata(project_dir: Path) -> dict:
    def run_git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=project_dir, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    dirty_files = run_git("status", "--porcelain").splitlines()
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_outputs(project_dir: Path, experiment_id: str, output_paths: list[Path]) -> tuple[Path, dict]:
    run_dir = project_dir / "csvFiles" / "experiments" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    archived = {}
    for source in output_paths:
        source = source.resolve()
        if not source.exists():
            continue
        destination = run_dir / source.name
        shutil.copy2(source, destination)
        archived[source.name] = {
            "source": str(source),
            "archived": str(destination.relative_to(project_dir)),
            "sha256": sha256_file(destination),
        }
    return run_dir, archived


def optional_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def add_experiment_id(frame, experiment_id: str):
    out = frame.copy()
    out.insert(0, "experiment_id", experiment_id)
    return out


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    threads = int(args.threads or default_threads())
    catboost_seeds = parse_seed_list(args.catboost_seeds)
    experiment_id = validate_experiment_id(args.experiment_id or default_experiment_id())
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))

    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    mpl_cache_dir = project_dir / ".matplotlib_cache"
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import ett_ensemble_pipeline as ens

    importlib.reload(ens)

    use_catboost = not args.no_catboost and not args.quick
    full_backtest = not args.no_backtest and not args.quick
    output_path = args.output if args.output.is_absolute() else project_dir / args.output
    csv_dir = project_dir / "csvFiles"
    csv_dir.mkdir(parents=True, exist_ok=True)
    git = git_metadata(project_dir)
    print(
        "local run profile:",
        {
            "experiment_id": experiment_id,
            "threads": threads,
            "catboost_gpu": bool(args.catboost_gpu and use_catboost),
            "catboost_seeds": catboost_seeds if use_catboost else (),
        },
    )

    result = ens.run_ensemble(
        plot=args.plot,
        full_backtest=full_backtest,
        use_lightgbm=not args.no_lightgbm,
        use_catboost=use_catboost,
        num_threads=threads,
        catboost_gpu=args.catboost_gpu and use_catboost,
        catboost_seeds=catboost_seeds,
        output_path=output_path,
    )

    candidate_path = csv_dir / "ensemble_candidates.csv"
    weights_path = csv_dir / "ensemble_weights.csv"
    lambda_path = csv_dir / "ensemble_lambda_search.csv"
    backtest_path = csv_dir / "ensemble_backtest.csv"
    horizon_summary_path = csv_dir / "ensemble_horizon_summary.csv"
    submission_distribution_path = csv_dir / "ensemble_submission_distribution.csv"

    candidate = add_experiment_id(result["candidate"], experiment_id)
    candidate.insert(1, "selected", candidate["name"] == result["predictor"]["candidate"])
    candidate.insert(2, "git_commit", git["commit"])
    candidate.insert(3, "catboost_seeds", ",".join(str(seed) for seed in catboost_seeds) if use_catboost else "")
    candidate.insert(4, "lambda_grid", ",".join(str(value) for value in ens.RESIDUAL_LAMBDA_GRID))
    candidate.insert(5, "threads", threads)
    candidate.insert(6, "catboost_gpu", bool(args.catboost_gpu and use_catboost))
    candidate.to_csv(candidate_path, index=False)
    add_experiment_id(result["weights"], experiment_id).to_csv(weights_path, index=False)
    add_experiment_id(result["lambda"], experiment_id).to_csv(lambda_path, index=False)
    add_experiment_id(result["backtest"], experiment_id).to_csv(backtest_path, index=False)
    add_experiment_id(result["horizon_summary"], experiment_id).to_csv(horizon_summary_path, index=False)
    add_experiment_id(result["submission_distribution"], experiment_id).to_csv(submission_distribution_path)

    output_paths = [
        output_path,
        candidate_path,
        weights_path,
        lambda_path,
        backtest_path,
        horizon_summary_path,
        submission_distribution_path,
        *result["candidate_submissions"].values(),
    ]
    run_dir, archived = archive_outputs(project_dir, experiment_id, output_paths)
    predictor = result["predictor"]
    manifest = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "command": sys.argv,
        "settings": {
            "threads": threads,
            "quick": bool(args.quick),
            "full_backtest": full_backtest,
            "use_lightgbm": not args.no_lightgbm,
            "use_catboost": use_catboost,
            "catboost_gpu": bool(args.catboost_gpu and use_catboost),
            "catboost_seeds": list(catboost_seeds) if use_catboost else [],
            "residual_lambda_grid": list(ens.RESIDUAL_LAMBDA_GRID),
        },
        "selected_predictor": {
            "name": predictor["name"],
            "kind": predictor["kind"],
            "candidate": predictor["candidate"],
            "lambda": optional_float(predictor.get("lambda")),
            "mse": float(predictor["metric"]["mse"]),
            "rmse": float(predictor["metric"]["rmse"]),
            "mae": float(predictor["metric"]["mae"]),
            "selection_reason": predictor["selection_reason"],
        },
        "public_score": None,
        "output_files": archived,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("selected predictor")
    print("=" * 80)
    print(result["predictor"])
    print("\nSaved:")
    print(output_path)
    print(candidate_path)
    print(weights_path)
    print(lambda_path)
    print(backtest_path)
    print(horizon_summary_path)
    print(submission_distribution_path)
    for path in result["candidate_submissions"].values():
        print(path)
    print(manifest_path)


if __name__ == "__main__":
    main()
