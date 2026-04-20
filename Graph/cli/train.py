from __future__ import annotations

import argparse

from training.pipeline import train_one_season


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--quarter", type=int, required=True)
    ap.add_argument("--ablation", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = train_one_season(args.year, args.quarter, args.ablation, args.seed)
    print(f"wrote {len(df)} rows for {args.year}Q{args.quarter} / {args.ablation}")


if __name__ == "__main__":
    main()
