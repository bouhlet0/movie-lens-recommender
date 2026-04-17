import numpy as np
import polars as pl


class BaseRecommender:
    supports_ranking: bool = True

    def fit(self, train_df: pl.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_idx: int, k: int) -> list[int]:
        raise NotImplementedError