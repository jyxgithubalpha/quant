from __future__ import annotations

import json
import os

import polars as pl

from data.io import load_raw_tables, wide_to_long
from domain.config import ExperimentConfig
from evaluation.metrics import get_metrics
from training.pipeline import train_one_season


def run_ablation(ablation: str, exp_cfg: ExperimentConfig, label_long, liquid_wide) -> dict:
    dfs = [train_one_season(y, q, ablation=ablation, exp_cfg=exp_cfg) for (y, q) in exp_cfg.quarters]
    combined = dfs[0] if len(dfs) == 1 else pl.concat(dfs)
    return get_metrics(combined, label_long, liquid_wide, money=exp_cfg.eval.money)


def main() -> None:
    exp_cfg = ExperimentConfig()
    _, label_wide, liquid_wide = load_raw_tables(exp_cfg.data)
    label_long = wide_to_long(label_wide, "label")

    results = {}
    for ablation in exp_cfg.ablations:
        print(f"\n=== {ablation} ===")
        metrics = run_ablation(ablation, exp_cfg, label_long, liquid_wide)
        results[ablation] = metrics
        print(metrics)

    os.makedirs(exp_cfg.results_dir, exist_ok=True)
    out_path = os.path.join(exp_cfg.results_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
