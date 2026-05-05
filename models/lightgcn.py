import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import Tensor
import pickle

from .base import BaseRecommender
from .utils import build_seen_items

def _build_seen_csr(
    seen: dict[int, set],
    n_users: int,
) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    for u, items in seen.items():
        indptr[u + 1] = len(items)
    np.cumsum(indptr, out=indptr)

    indices = np.empty(indptr[-1], dtype=np.int64)
    for u, items in seen.items():
        lo, hi = indptr[u], indptr[u + 1]
        indices[lo:hi] = np.sort(
            np.fromiter(items, dtype=np.int64, count=len(items))
        )
    return indices, indptr


def _sample_negatives_vectorized(
    u_batch: np.ndarray,
    seen_csr_indices: np.ndarray,
    seen_csr_indptr: np.ndarray,
    n_items: int,
    max_retries: int = 10,
) -> np.ndarray:
    n_batch = np.random.randint(0, n_items, size=len(u_batch), dtype=np.int64)
    needs_resample = np.ones(len(u_batch), dtype=bool)

    for _ in range(max_retries):
        if not needs_resample.any():
            break
        idx   = np.where(needs_resample)[0]
        u_sub = u_batch[idx]
        cand  = n_batch[idx]

        lo = seen_csr_indptr[u_sub]
        hi = seen_csr_indptr[u_sub + 1]

        collision = np.zeros(len(idx), dtype=bool)
        for j in range(len(idx)):
            s = seen_csr_indices[lo[j]:hi[j]]
            if len(s):
                ins = np.searchsorted(s, cand[j])
                collision[j] = ins < len(s) and s[ins] == cand[j]

        still_bad = idx[collision]
        n_batch[still_bad] = np.random.randint(
            0, n_items, size=len(still_bad), dtype=np.int64
        )
        needs_resample[still_bad] = True
        needs_resample[idx[~collision]] = False

    return n_batch


class _LightGCNCore(nn.Module):
    """
    LightGCN core following He et al. 2020.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        n_layers: int,
        train_df: pl.DataFrame,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.n_users    = n_users
        self.n_items    = n_items
        self.n_layers   = n_layers

        self.E0 = nn.Embedding(n_users + n_items, embedding_dim)
        nn.init.normal_(self.E0.weight, std=0.1)

        # Build and register normalised adjacency as buffer
        # (moves to correct device with .to(device), not updated by optimiser)
        adj = self._build_norm_adj(train_df, n_users, n_items, device)
        self.register_buffer("norm_adj", adj)

    @staticmethod
    def _build_norm_adj(
        train_df: pl.DataFrame,
        n_users: int,
        n_items: int,
        device: torch.device,
    ) -> Tensor:
        users = train_df["user_idx"].to_numpy().astype(np.int32)
        items = train_df["item_idx"].to_numpy().astype(np.int32)

        n = n_users + n_items

        R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
        R[users, items] = 1.0
        R = R.tolil()

        adj = sp.lil_matrix((n, n), dtype=np.float32)
        adj[:n_users, n_users:] = R
        adj[n_users:, :n_users] = R.T
        adj = adj.tocsr()

        # D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum + 1e-9, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D = sp.diags(d_inv_sqrt)
        norm_adj = D @ adj @ D

        # Convert to PyTorch sparse COO
        coo = norm_adj.tocoo().astype(np.float32)
        indices = torch.tensor(
            np.vstack([coo.row, coo.col]), dtype=torch.long
        )
        values = torch.tensor(coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, (n, n)
        ).coalesce().to(device)

    def propagate(self) -> Tensor:
        x = self.E0.weight
        all_x = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            all_x.append(x)
        return torch.stack(all_x, dim=1).mean(dim=1)

    def forward(
        self,
        users: Tensor,
        pos_items: Tensor,
        neg_items: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        mean_emb = self.propagate()
        final_user = mean_emb[:self.n_users]
        final_item = mean_emb[self.n_users:]

        users_emb = final_user[users]
        pos_emb   = final_item[pos_items]
        neg_emb   = final_item[neg_items]

        u0 = self.E0(users)
        p0 = self.E0(pos_items + self.n_users)
        n0 = self.E0(neg_items + self.n_users)

        return users_emb, pos_emb, neg_emb, u0, p0, n0

class LightGCNModel(BaseRecommender):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network
    for Recommendation. He et al., 2020.
    """

    supports_ranking: bool = True

    _user_factors: np.ndarray | None
    _item_factors: np.ndarray | None
    _seen: dict[int, set]

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 128,
        n_layers: int = 2,
        lr: float = 1e-5,
        reg_weight: float = 1e-5,
        n_epochs: int = 50,
        batch_size: int = 32768,
        device: str | None = None,
    ) -> None:
        self.n_users       = n_users
        self.n_items       = n_items
        self.embedding_dim = embedding_dim
        self.n_layers      = n_layers
        self.lr            = lr
        self.reg_weight    = reg_weight
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.loss_history: list[float] = []

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._user_factors  = None
        self._item_factors  = None
        self._seen          = {}

    def fit(self, train_df: pl.DataFrame) -> None:
        print("Building seen dict...")
        self._seen = build_seen_items(train_df)

        print("Building CSR seen structure...")
        seen_csr_indices, seen_csr_indptr = _build_seen_csr(
            self._seen, self.n_users
        )

        print("Building model and adjacency matrix...")
        model = _LightGCNCore(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
            train_df=train_df,
            device=self.device,
        ).to(self.device)

        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        users_all       = train_df["user_idx"].to_numpy().astype(np.int64)
        items_all       = train_df["item_idx"].to_numpy().astype(np.int64)
        n_interactions  = len(users_all)

        for epoch in range(self.n_epochs):
            model.train()

            perm        = np.random.permutation(n_interactions)
            users_epoch = users_all[perm]
            items_epoch = items_all[perm]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n_interactions, self.batch_size):
                end     = min(start + self.batch_size, n_interactions)
                u_batch = users_epoch[start:end]
                p_batch = items_epoch[start:end]

                n_batch = _sample_negatives_vectorized(
                    u_batch, seen_csr_indices, seen_csr_indptr, self.n_items
                )

                u_t = torch.tensor(u_batch, dtype=torch.long, device=self.device)
                p_t = torch.tensor(p_batch, dtype=torch.long, device=self.device)
                n_t = torch.tensor(n_batch, dtype=torch.long, device=self.device)

                users_emb, pos_emb, neg_emb, u0, p0, n0 = model(u_t, p_t, n_t)

                pos_scores = (users_emb * pos_emb).sum(dim=1)
                neg_scores = (users_emb * neg_emb).sum(dim=1)
                bpr_loss   = -torch.log(
                    torch.sigmoid(pos_scores - neg_scores) + 1e-8
                ).mean()

                reg = (
                    u0.norm(2).pow(2) +
                    p0.norm(2).pow(2) +
                    n0.norm(2).pow(2)
                ) / float(len(u_batch))

                loss = bpr_loss + self.reg_weight * reg

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            print(f"  Epoch {epoch+1:>3}/{self.n_epochs}  loss={avg_loss:.4f}")

        # Extract final embeddings for inference
        model.eval()
        with torch.no_grad():
            mean_emb = model.propagate()

        self._user_factors = mean_emb[:self.n_users].cpu().numpy().astype(np.float32)
        self._item_factors = mean_emb[self.n_users:].cpu().numpy().astype(np.float32)

    def predict(self, eval_df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "LightGCNModel is a ranking model; use recommend() instead."
        )

    def recommend(self, user_idx: int, k: int) -> list[int]:
        if self._user_factors is None or user_idx >= self.n_users:
            return []

        scores = (
            self._item_factors @ self._user_factors[user_idx]
        ).astype(np.float32)

        seen = self._seen.get(user_idx)
        if seen:
            scores[list(seen)] = -np.inf

        top_k = min(k, self.n_items - len(seen or []))
        if top_k <= 0:
            return []

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return top_indices.tolist()
    
    @classmethod
    def load(cls, path: str) -> "LightGCNModel":
        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls(
            n_users=data["n_users"],
            n_items=data["n_items"],
        )

        model._user_factors = data["user_factors"]
        model._item_factors = data["item_factors"]
        model._seen = data["seen"]

        return model