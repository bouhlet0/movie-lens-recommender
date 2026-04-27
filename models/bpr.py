import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking

from .base import BaseRecommender
from .utils import build_seen_items


class BPRModel(BaseRecommender):
    """
    Bayesian Personalised Ranking via the implicit library.
    """

    supports_ranking: bool = True

    _user_factors: np.ndarray | None
    _item_factors: np.ndarray | None
    _seen: dict[int, set[int]]
    _n_items: int

    def __init__(
        self,
        factors: int = 128,
        iterations: int = 100,
        learning_rate: float = 0.05,
        regularization: float = 0.001,
        use_gpu: bool | None = None,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization

        if use_gpu is None:
            try:
                import cupy
                use_gpu = True
                print("Using GPU with cupy")
            except ImportError:
                use_gpu = False
        self.use_gpu = use_gpu

        self._model = BayesianPersonalizedRanking(
            factors=factors,
            iterations=iterations,
            learning_rate=learning_rate,
            regularization=regularization,
            use_gpu=use_gpu,
        )
        self._user_factors = None
        self._item_factors = None
        self._seen = {}
        self._n_items = 0

    def fit(self, train_df: pl.DataFrame, implicit_matrix: sp.csr_matrix) -> None:
        self._seen = build_seen_items(train_df)
        self._n_items = implicit_matrix.shape[1]
        self._model.fit(implicit_matrix.T.tocsr())

        uf  = self._model.user_factors
        itf = self._model.item_factors
        if self.use_gpu:
            uf  = uf.to_numpy()
            itf = itf.to_numpy()

        # Same implicit quirk as ALS
        self._user_factors = itf.astype(np.float32)
        self._item_factors = uf.astype(np.float32)

    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "BPRModel is a ranking model and does not support rating prediction."
        )

    def recommend(self, user_idx: int, k: int) -> list[int]:
        if self._user_factors is None or user_idx >= len(self._user_factors):
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