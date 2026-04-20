"""
Industry grouping and prediction score neutralization module

A-share industry classification rules (by stock code prefix):
  000xxx, 001xxx → SZ_MAIN   (Shenzhen Stock Exchange Main Board)
  002xxx         → SME        (Small and Medium Enterprise Board, now merged into SZ Main Board, but historically preserved)
  300xxx         → CHINEXT    (ChiNext Board)
  600xxx, 601xxx, 603xxx, 605xxx → SH_MAIN (Shanghai Stock Exchange Main Board)
  688xxx         → STAR       (STAR Market)
  Others         → OTHER

Neutralization: Subtract prediction score median within each (date, industry) group to eliminate industry rotation effects.
"""

import polars as pl


# ============================================================
# Industry identification (polars when/then chain)
# ============================================================
def _board_expr() -> pl.Expr:
    """
    Returns a polars expression that maps industry identifiers based on Code column prefix.
    """
    return (
        pl.when(pl.col("Code").str.starts_with("300")).then(pl.lit("CHINEXT"))
        .when(pl.col("Code").str.starts_with("688")).then(pl.lit("STAR"))
        .when(pl.col("Code").str.starts_with("002")).then(pl.lit("SME"))
        .when(pl.col("Code").str.starts_with("000") | pl.col("Code").str.starts_with("001"))
        .then(pl.lit("SZ_MAIN"))
        .when(
            pl.col("Code").str.starts_with("600")
            | pl.col("Code").str.starts_with("601")
            | pl.col("Code").str.starts_with("603")
            | pl.col("Code").str.starts_with("605")
        ).then(pl.lit("SH_MAIN"))
        .otherwise(pl.lit("OTHER"))
        .alias("board")
    )


# ============================================================
# Prediction score neutralization
# ============================================================
def neutralize_scores(score_df: pl.DataFrame) -> pl.DataFrame:
    """
    Subtract prediction score median within each (date, industry) group to achieve industry neutralization.

    Parameters
    ----------
    score_df : pl.DataFrame, containing ['date', 'Code', 'score'] columns

    Returns
    -------
    Neutralized pl.DataFrame, same format
    """
    df = score_df.with_columns(_board_expr())

    # Calculate group medians and join back
    medians = df.group_by(["date", "board"]).agg(
        pl.col("score").median().alias("_median")
    )
    df = df.join(medians, on=["date", "board"], how="left")
    df = df.with_columns(
        (pl.col("score") - pl.col("_median")).alias("score")
    ).drop("board", "_median")

    return df
