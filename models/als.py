import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from .base import BaseRecommender
from .utils import build_seen_items


class ALSModel(BaseRecommender):
    """
    Matrix Factorization via Alternating Least Squares on implicit feedback.
    Confidence matrix: c_ui = 1 + alpha * rating_ui.
    Operates on a user-item confidence CSR matrix produced by
    data.to_implicit_matrix().
    """

    supports_ranking: bool = True

    def __init__(
        self,
        factors: int = 50,
        iterations: int = 20,
        regularization: float = 0.01,
        use_gpu: bool = False,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization

        self._model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            use_gpu=use_gpu,
        )
        self._user_factors: np.ndarray = None
        self._item_factors: np.ndarray = None
        self._seen: dict[int, set[int]] = {}
        self._n_items: int = 0

    def fit(self, train_df: pl.DataFrame, implicit_matrix: sp.csr_matrix) -> None:
        self._seen = build_seen_items(train_df)
        self._n_items = implicit_matrix.shape[1]

        self._model.fit(implicit_matrix.T.tocsr())

        self._user_factors = self._model.user_factors
        self._item_factors = self._model.item_factors

    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError("ALSModel is a ranking model and does not support rating prediction.")

    def recommend(self, user_idx: int, k: int) -> list[int]:
        if user_idx >= len(self._user_factors):
            return []

        scores = (self._item_factors @ self._user_factors[user_idx]).astype(np.float32)

        seen = self._seen.get(user_idx)
        if seen:
            scores[list(seen)] = -np.inf

        top_k = min(k, self._n_items - len(seen or []))
        if top_k <= 0:
            return []

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return top_indices.tolist()