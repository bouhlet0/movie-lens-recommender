import numpy as np
import polars as pl
from .base import BaseRecommender
from .utils import build_seen_items


class UserMeanModel(BaseRecommender):
    """
    Predicts per-user mean rating.
    Falls back to global mean for items not seen in training.
    Doesn't support ranking.
    """
    supports_ranking: bool = False
    
    def __init__(self) -> None:
        self.global_mean: float = 0.0
        self._user_means: dict[int, float] = {}
        self._seen: dict[int, set[int]] = {}
            
    def fit(self, train_df :pl.DataFrame) -> None:
        self.global_mean= train_df["rating"].mean()
        self._user_means = dict(
            train_df
            .group_by("user_idx")
            .agg(pl.col("rating").mean.alias("mean"))
            .iter_rows()
        )
        self._seen = build_seen_items(train_df)
        
    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        return np.array(
            [
                self._user_means.get(item, self.global_mean)
                for item in eval_df["user_idx"].to_list()
            ],
            dtype=np.float32,
        )
        
    def recommend(self, user_idx: int, k: int) -> list[int]:
        raise NotImplementedError("ItemMeanModel doesn't support ranking.")