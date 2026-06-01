from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ETTh1 ensemble pipeline locally.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("submit_ensemble.csv"))
    parser.add_argument("--plot", action="store_true", help="Show validation RMSE plot.")
    parser.add_argument("--quick", action="store_true", help="Skip rolling backtest and CatBoost for a fast smoke test.")
    parser.add_argument("--no-backtest", action="store_true", help="Skip rolling backtest.")
    parser.add_argument("--no-lightgbm", action="store_true", help="Disable LightGBM residual model.")
    parser.add_argument("--no-catboost", action="store_true", help="Disable CatBoost residual model.")
    parser.add_argument("--threads", type=int, default=None, help="CPU worker threads. Default keeps two logical threads free.")
    parser.add_argument("--catboost-gpu", action="store_true", help="Try CatBoost on RTX GPU, then fall back to CPU if unavailable.")
    parser.add_argument("--catboost-seeds", default="42,1337,2026", help="Comma-separated CatBoost seeds for seed ensembling.")
    return parser.parse_args()


def default_threads() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("--catboost-seeds must contain at least one integer seed.")
    return seeds


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    threads = int(args.threads or default_threads())
    catboost_seeds = parse_seed_list(args.catboost_seeds)
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
    print(
        "local run profile:",
        {"threads": threads, "catboost_gpu": bool(args.catboost_gpu and use_catboost), "catboost_seeds": catboost_seeds if use_catboost else ()},
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

    result["candidate"].to_csv(project_dir / "ensemble_candidates.csv", index=False)
    result["weights"].to_csv(project_dir / "ensemble_weights.csv", index=False)
    result["lambda"].to_csv(project_dir / "ensemble_lambda_search.csv", index=False)
    result["backtest"].to_csv(project_dir / "ensemble_backtest.csv", index=False)
    result["horizon_summary"].to_csv(project_dir / "ensemble_horizon_summary.csv", index=False)
    result["submission_distribution"].to_csv(project_dir / "ensemble_submission_distribution.csv")

    print("\n" + "=" * 80)
    print("selected predictor")
    print("=" * 80)
    print(result["predictor"])
    print("\nSaved:")
    print(output_path)
    print(project_dir / "ensemble_candidates.csv")
    print(project_dir / "ensemble_weights.csv")
    print(project_dir / "ensemble_lambda_search.csv")
    print(project_dir / "ensemble_backtest.csv")
    print(project_dir / "ensemble_horizon_summary.csv")
    print(project_dir / "ensemble_submission_distribution.csv")
    for path in result["candidate_submissions"].values():
        print(path)


if __name__ == "__main__":
    main()
