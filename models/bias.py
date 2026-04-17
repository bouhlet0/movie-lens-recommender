import numpy as np
import polars as pl
from .base import BaseRecommender
from .utils import build_seen_items


class BiasModel(BaseRecommender):
    """
    Predicts r_ui = μ + b_u + b_i where biases are fit via ALS with L2 regularisation.
    First real structured model, should meaningfully beat user/item mean baselines.
    Does not support ranking.
    """
    supports_ranking: bool = False

    def __init__(self, n_epochs: int = 10, lambda_: float = 25.0) -> None:
        self.n_epochs = n_epochs
        self.lambda_ = lambda_
        self.global_mean: float = 0.0
        self._user_biases: dict[int, float] = {}
        self._item_biases: dict[int, float] = {}
        self._seen: dict[int, set[int]] = {}

    def fit(self, train_df: pl.DataFrame) -> None:
        self.global_mean = train_df["rating"].mean()
        self._seen = build_seen_items(train_df)

        # Pre-aggregate once, reused every epoch
        # ratings residual relative to global mean
        df = train_df.with_columns(
            (pl.col("rating") - self.global_mean).alias("residual")
        )

        user_groups = {
            row["user_idx"]: (
                np.array(row["item_idxs"], dtype=np.int32),
                np.array(row["residuals"], dtype=np.float32),
            )
            for row in (
                df.group_by("user_idx")
                .agg([
                    pl.col("item_idx").alias("item_idxs"),
                    pl.col("residual").alias("residuals"),
                ])
                .iter_rows(named=True)
            )
        }

        item_groups = {
            row["item_idx"]: (
                np.array(row["user_idxs"], dtype=np.int32),
                np.array(row["residuals"], dtype=np.float32),
            )
            for row in (
                df.group_by("item_idx")
                .agg([
                    pl.col("user_idx").alias("user_idxs"),
                    pl.col("residual").alias("residuals"),
                ])
                .iter_rows(named=True)
            )
        }

        # Initialise biases to zero
        b_u: dict[int, float] = {u: 0.0 for u in user_groups}
        b_i: dict[int, float] = {i: 0.0 for i in item_groups}

        for epoch in range(self.n_epochs):
            # Fix b_u, solve for b_i
            for item_idx, (user_idxs, residuals) in item_groups.items():
                bu_vec = np.array([b_u.get(u, 0.0) for u in user_idxs], dtype=np.float32)
                b_i[item_idx] = (residuals - bu_vec).sum() / (self.lambda_ + len(residuals))

            # Fix b_i, solve for b_u
            for user_idx, (item_idxs, residuals) in user_groups.items():
                bi_vec = np.array([b_i.get(i, 0.0) for i in item_idxs], dtype=np.float32)
                b_u[user_idx] = (residuals - bi_vec).sum() / (self.lambda_ + len(residuals))

        self._user_biases = b_u
        self._item_biases = b_i

    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        return np.array(
            [
                self.global_mean
                + self._user_biases.get(row["user_idx"], 0.0)
                + self._item_biases.get(row["item_idx"], 0.0)
                for row in eval_df.select(["user_idx", "item_idx"]).iter_rows(named=True)
            ],
            dtype=np.float32,
        )

    def recommend(self, user_idx: int, k: int) -> list[int]:
        raise NotImplementedError("BiasModel doesn't support ranking.")