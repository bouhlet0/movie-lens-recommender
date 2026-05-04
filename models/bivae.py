import ast
import numpy as np
import polars as pl
import torch
from cornac.models import BiVAECF
from cornac.data import Dataset as CornacDataset
# from cornac.utils import Callbacks

from .base import BaseRecommender
from .utils import build_seen_items


class _LossHistoryCallback:
    """
    Cornac-compatible callback that records per-epoch loss values.
    Cornac calls on_epoch_end(epoch, logs) after each epoch.
    """
    def __init__(self) -> None:
        self.loss_i_history: list[float] = []
        self.loss_u_history: list[float] = []

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        self.loss_i_history.append(float(logs.get("loss_i", float("nan"))))
        self.loss_u_history.append(float(logs.get("loss_u", float("nan"))))


class BiVAEModel(BaseRecommender):
    """
    Bilateral Variational Autoencoder for Collaborative Filtering.
    Salah et al., 2021. Wrapped from Cornac's implementation.

    Likelihood modes:
    - 'bern': Bernoulli : binary interactions (rating >= threshold only)
    - 'pois': Poisson   : all interactions with raw ratings as counts
              Note: star ratings are not true Poisson counts but this
              likelihood is empirically more stable on sparse data.

    Loss history is stored in self.loss_i_history and self.loss_u_history
    after fit() completes.
    """

    supports_ranking: bool = True

    _user_factors: np.ndarray | None
    _item_factors: np.ndarray | None
    _seen: dict[int, set[int]]
    _n_items: int
    _cornac_model: BiVAECF | None
    _user_idx_to_cornac: dict[int, int] | None
    _item_idx_to_cornac: dict[int, int] | None
    _cornac_to_item_idx: dict[int, int] | None
    _cornac_indices: np.ndarray | None
    _your_indices: np.ndarray | None

    def __init__(
        self,
        k: int = 64,
        encoder_structure: list[int] | str | None = None,
        act_fn: str = "tanh",
        likelihood: str = "bern",
        n_epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        threshold: float = 4.0,
        seed: int = 42,
    ) -> None:
        self.k             = k
        self.act_fn        = act_fn
        self.likelihood    = likelihood
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.threshold     = threshold
        self.seed          = seed

        if encoder_structure is None:
            self.encoder_structure = [256]
        elif isinstance(encoder_structure, str):
            self.encoder_structure = ast.literal_eval(encoder_structure)
        else:
            self.encoder_structure = encoder_structure

        self._user_factors       = None
        self._item_factors       = None
        self._seen               = {}
        self._n_items            = 0
        self._cornac_model       = None
        self._user_idx_to_cornac = None
        self._item_idx_to_cornac = None
        self._cornac_to_item_idx = None
        self._cornac_indices     = None
        self._your_indices       = None

        # Loss history populated after fit()
        self.loss_i_history: list[float] = []
        self.loss_u_history: list[float] = []

    def _to_cornac_dataset(self, train_df: pl.DataFrame) -> CornacDataset:
        if self.likelihood == "bern":
            filtered = train_df.filter(pl.col("rating") >= self.threshold)
            uir = list(zip(
                filtered["user_idx"].to_list(),
                filtered["item_idx"].to_list(),
                [1.0] * len(filtered),
            ))
        else:
            uir = list(zip(
                train_df["user_idx"].to_list(),
                train_df["item_idx"].to_list(),
                train_df["rating"].to_list(),
            ))
        return CornacDataset.from_uir(uir, seed=self.seed)

    def fit(self, train_df: pl.DataFrame) -> None:
        self._seen    = build_seen_items(train_df)
        self._n_items = train_df["item_idx"].n_unique()

        print("  Converting to Cornac dataset...")
        cornac_data = self._to_cornac_dataset(train_df)

        self._user_idx_to_cornac = {
            int(k): v for k, v in cornac_data.uid_map.items()
        }
        self._item_idx_to_cornac = {
            int(k): v for k, v in cornac_data.iid_map.items()
        }
        self._cornac_to_item_idx = {
            v: k for k, v in self._item_idx_to_cornac.items()
        }
        self._cornac_indices = np.array(
            list(self._cornac_to_item_idx.keys()), dtype=np.int64
        )
        self._your_indices = np.array(
            list(self._cornac_to_item_idx.values()), dtype=np.int64
        )

        print(f"  Your users: {train_df['user_idx'].n_unique():,}  "
              f"Cornac users: {cornac_data.num_users:,}  "
              f"Dropped: {train_df['user_idx'].n_unique() - cornac_data.num_users:,}")
        print(f"  Your items: {train_df['item_idx'].n_unique():,}  "
              f"Cornac items: {cornac_data.num_items:,}  "
              f"Dropped: {train_df['item_idx'].n_unique() - cornac_data.num_items:,}")

        self._cornac_model = BiVAECF(
            k=self.k,
            encoder_structure=self.encoder_structure,
            act_fn=self.act_fn,
            likelihood=self.likelihood,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
            use_gpu=torch.cuda.is_available(),
            verbose=True,
        )

        print("  Training BiVAE...")
        self._cornac_model.fit(cornac_data)

        self._user_factors = self._cornac_model.get_user_vectors().astype(np.float32)
        self._item_factors = self._cornac_model.get_item_vectors().astype(np.float32)

        # Extract loss history from Cornac's internal tracking
        # Cornac stores per-epoch losses in the model after fit
        if hasattr(self._cornac_model, "losses"):
            losses = self._cornac_model.losses
            if isinstance(losses, dict):
                self.loss_i_history = [float(v) for v in losses.get("loss_i", [])]
                self.loss_u_history = [float(v) for v in losses.get("loss_u", [])]
            elif isinstance(losses, list):
                self.loss_i_history = [float(v) for v in losses]
        else:
            print("  Note: loss history not available from Cornac model.")

        print(f"  user_factors shape: {self._user_factors.shape}")
        print(f"  item_factors shape: {self._item_factors.shape}")

    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "BiVAEModel is a ranking model and does not support rating prediction."
        )

    def recommend(self, user_idx: int, k: int) -> list[int]:
        if self._user_factors is None or self._user_idx_to_cornac is None:
            return []

        cornac_user = self._user_idx_to_cornac.get(user_idx)
        if cornac_user is None:
            return []

        user_vec      = self._user_factors[cornac_user]
        scores_cornac = (self._item_factors @ user_vec).astype(np.float32)

        scores = np.full(self._n_items, -np.inf, dtype=np.float32)
        scores[self._your_indices] = scores_cornac[self._cornac_indices]

        seen = self._seen.get(user_idx)
        if seen:
            scores[list(seen)] = -np.inf

        n_valid = int(np.isfinite(scores).sum())
        top_k   = min(k, n_valid)
        if top_k <= 0:
            return []

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        top_indices = top_indices[np.isfinite(scores[top_indices])]

        return top_indices.tolist()