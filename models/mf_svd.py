import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

from .base import BaseRecommender
from .utils import build_seen_items


class MFSVDModel(BaseRecommender):
    """
    Matrix Factorization via Truncated SVD.
    Predicts r_ui = μ + (u_factors * i_factors).
    Supports both rating prediction and ranking.
    
    Cold-Start Behavior:
    - predict(): Falls back to the global mean for unseen users/items.
    - recommend(): Returns an empty list for unseen users.
    """

    supports_ranking: bool = True

    def __init__(
        self,
        k: int = 50,
        min_rating: float = 0.5,
        max_rating: float = 5.0,
    ) -> None:
        self.k = k
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.global_mean: float = 0.0
        self._user_factors: np.ndarray = None
        self._item_factors: np.ndarray = None
        self._seen: dict[int, set[int]] = {}
        self._n_users: int = 0
        self._n_items: int = 0

    def fit(self, train_df: pl.DataFrame) -> None:
        self.global_mean = train_df["rating"].mean()
        self._seen = build_seen_items(train_df)

        self._n_users = train_df.select(pl.col("user_idx").max()).item() + 1
        self._n_items = train_df.select(pl.col("item_idx").max()).item() + 1

        # Prevent invalid SVD rank
        max_k = min(self._n_users, self._n_items) - 1
        if max_k < 1:
            raise ValueError("Dataset dimensions too small to perform SVD.")
        valid_k = max(1, min(self.k, max_k))

        row = train_df["user_idx"].to_numpy()
        col = train_df["item_idx"].to_numpy()
        data = train_df["rating"].to_numpy().astype(np.float32) - self.global_mean

        sparse_ratings = coo_matrix(
            (data, (row, col)),
            shape=(self._n_users, self._n_items),
        ).tocsr()

        # Truncated SVD
        U, S, Vt = svds(sparse_ratings, k=valid_k)
        U = U[:, ::-1]
        S = S[::-1]
        Vt = Vt[::-1, :]

        s_sqrt = np.sqrt(S).astype(np.float32)
        self._user_factors = (U * s_sqrt).astype(np.float32)
        self._item_factors = (Vt.T * s_sqrt).astype(np.float32)

    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        users = eval_df["user_idx"].to_numpy()
        items = eval_df["item_idx"].to_numpy()

        valid_mask = (users < self._n_users) & (items < self._n_items)
        preds = np.full(len(eval_df), self.global_mean, dtype=np.float32)

        if np.any(valid_mask):
            valid_users = users[valid_mask]
            valid_items = items[valid_mask]

            interactions = np.einsum(
                "ij,ij->i",
                self._user_factors[valid_users],
                self._item_factors[valid_items],
            )

            preds[valid_mask] += interactions

        if self.min_rating is not None and self.max_rating is not None:
            preds = np.clip(preds, self.min_rating, self.max_rating)

        return preds

    def recommend(self, user_idx: int, k: int) -> list[int]:

        if user_idx >= self._n_users:
            return []

        scores = (self._item_factors @ self._user_factors[user_idx]).astype(np.float32)
        scores += self.global_mean

        seen = self._seen.get(user_idx)
        if seen:
            scores[list(seen)] = -np.inf

        top_k = min(k, self._n_items - len(seen or []))
        if top_k <= 0:
            return []

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return top_indices.tolist()