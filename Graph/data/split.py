from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from dateutil.relativedelta import relativedelta


def quarter_range(year: int, quarter: int) -> tuple[datetime, datetime]:
    start = datetime(year, (quarter - 1) * 3 + 1, 1)
    end = datetime(year + (quarter == 4), 1 if quarter == 4 else quarter * 3 + 1, 1) - timedelta(days=1)
    return start, end


def build_season_splits(
    fac_long: pl.DataFrame,
    year: int,
    quarter: int,
    valid_quarters: int = 2,
    gap_days: int = 10,
) -> tuple[list, list, list]:
    test_start, test_end = quarter_range(year, quarter)
    valid_end = test_start - timedelta(days=gap_days + 1)
    valid_start = test_start - relativedelta(months=valid_quarters * 3)
    train_end = valid_start - timedelta(days=gap_days + 1)
    train_start = fac_long["date"].min()

    all_dates = sorted(fac_long["date"].unique().to_list())
    pick = lambda a, b: [d for d in all_dates if a <= d <= b]
    return pick(train_start, train_end), pick(valid_start, valid_end), pick(test_start, test_end)
