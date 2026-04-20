"""Load and align 6 bench scoring files + labels into a unified long table."""

import polars as pl
from config import BENCH_PATHS, LABEL_PATH, LIQUID_PATH


def load_wide_feather(path: str, date_col: str = "date") -> pl.LazyFrame:
    """Read wide feather file and unpivot to long format (date, code, score)."""
    df = pl.read_ipc(path, memory_map=False)
    return df.unpivot(
        index=[date_col],
        variable_name="code",
        value_name="score",
    ).lazy()


def load_all_benches() -> pl.DataFrame:
    """Load all 6 bench files and join into (date, code, bench1..bench6)."""
    frames = []
    for name, path in BENCH_PATHS.items():
        lf = load_wide_feather(path, date_col="date")
        lf = lf.rename({"score": name}).drop_nulls(subset=[name])
        frames.append(lf)

    # Progressive join on (date, code) — inner join keeps only common stocks/dates
    result = frames[0]
    for lf in frames[1:]:
        result = result.join(lf, on=["date", "code"], how="inner")

    return result.collect()


def load_labels() -> pl.DataFrame:
    """Load label file (wide: index × stock codes) → long (date, code, label)."""
    df = pl.read_ipc(LABEL_PATH, memory_map=False)
    return df.unpivot(
        index=["index"],
        variable_name="code",
        value_name="label",
    ).rename({"index": "date"}).drop_nulls(subset=["label"])


def load_liquidity() -> pl.DataFrame:
    """Load liquidity file (wide: index × stock codes) → long (date, code, can_trade)."""
    df = pl.read_ipc(LIQUID_PATH, memory_map=False)
    return df.unpivot(
        index=["index"],
        variable_name="code",
        value_name="can_trade",
    ).rename({"index": "date"})


def load_data() -> pl.DataFrame:
    """Load benches + labels + liquidity, return merged DataFrame."""
    benches = load_all_benches()
    labels = load_labels()
    liquidity = load_liquidity()
    merged = benches.join(labels.lazy().collect(), on=["date", "code"], how="inner")
    merged = merged.join(liquidity, on=["date", "code"], how="left")
    return merged.sort(["date", "code"])


if __name__ == "__main__":
    df = load_data()
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"Stocks per date: {df.group_by('date').len().select('len').mean().item():.0f}")
    print(df.head(5))
