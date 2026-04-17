import polars as pl


def build_seen_items(train_df: pl.DataFrame) -> dict[int, set[int]]:
    """
    Returns {user_idx: set of item_idxs} for all interactions in train_df.
    Used by every model's recommend() to exclude already-seen items.
    """
    seen: dict[int, set[int]] = {}
    for row in (
        train_df
        .group_by("user_idx")
        .agg(pl.col("item_idx").alias("items"))
        .iter_rows(named=True)
    ):
        seen[row["user_idx"]] = set(row["items"])
    return seen