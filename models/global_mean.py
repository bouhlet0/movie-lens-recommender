import numpy as np
import polars as pl
from .base import BaseRecommender
from .utils import build_seen_items


class GlobalMeanModel (BaseRecommender):
    """
    Predicts global mean rating for every user-item pair.
    Serves as floor baseline, every subsequent model should be better.
    Doesn't support ranking..
    """
    supports_ranking: bool = False
    
    def __init__(self) -> None:
        self.global_mean: float = 0.0
        self._seen: dict[int, set[int]] = {}
        
    def fit(self, train_df: pl.DataFrame) -> None:
        self.global_mean = train_df["Rating"].mean()
        self._seen = build_seen_items(train_df)
        
    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        return np.full(len(eval_df), self.global_mean, dtype=np.float32)
    
    def recommend(self, user_idx: int, k: int) -> list[int]:
        raise NotImplementedError("GlobalMeanModel doesn't support ranking.")