import numpy as np
import polars as pl
from .base import BaseRecommender
from .utils import build_seen_items


class PopularityModel(BaseRecommender):
    """
    Recommends the globally most interacted-with items, ranked by interaction count.
    Non-personalised, every user gets the same list minus their seen items.
    First ranking baseline, sets the floor for all ranking metrics.
    Does not support rating prediction.
    """
    supports_ranking: bool = True

    def __init__(self) -> None:
        self._popular_items: list[int] = []
        self._seen: dict[int, set[int]] = {}

    def fit(self, train_df: pl.DataFrame) -> None:
        self._popular_items = (
            train_df
            .group_by("item_idx")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .select("item_idx")
            ["item_idx"]
            .to_list()
        )
        self._seen = build_seen_items(train_df)

    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError("PopularityModel doesn't support rating prediction.")

    def recommend(self, user_idx: int, k: int) -> list[int]:
        seen = self._seen.get(user_idx, set())
        recs = []
        for item in self._popular_items:
            if item not in seen:
                recs.append(item)
            if len(recs) == k:
                break
        return recs