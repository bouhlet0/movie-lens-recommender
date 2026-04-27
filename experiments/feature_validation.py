import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import scipy.sparse as sp

from data import build_dataset
from evaluate import evaluate_ranking_model
from models.base import BaseRecommender
from models.popularity import PopularityModel

from features.item_features import (
    build_item_feature_matrix,
    normalize_rows,
    compute_topk_cosine_similarity,
)


class ItemKNNRecommender(BaseRecommender):
    """
    Base Item-item cosine similarity recommender using item features.
    """

    supports_ranking: bool = True

    def __init__(self, item_sim: np.ndarray):
        self.item_sim = item_sim
        self._seen = {}
        self._n_items = item_sim.shape[0]

    def fit(self, train_df: pl.DataFrame) -> None:
        from models.utils import build_seen_items
        self._seen = build_seen_items(train_df)

    def predict(self, eval_df: pl.DataFrame):
        raise NotImplementedError

    def recommend(self, user_idx: int, k: int) -> list[int]:
        seen = self._seen.get(user_idx)
        if not seen:
            return []

        scores = np.zeros(self._n_items, dtype=np.float32)

        for item in seen:
            scores += self.item_sim.getrow(item).toarray().ravel()

        scores[list(seen)] = -np.inf

        top_k = min(k, self._n_items - len(seen))
        if top_k <= 0:
            return []

        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(-scores[idx])]
        return idx.tolist()

def main() -> None:
    print("Building LLN dataset...")
    ds = build_dataset(split="leave_last_n", k=10, lln_n=2)

    print("Building item features...")
    X_items, genre2idx = build_item_feature_matrix(
        ds.movies_df,
        ds.item2idx,
    )

    print(f"  Item feature matrix: {X_items.shape}")

    print("Normalizing features...")
    X_norm = normalize_rows(X_items)

    print("Computing top-K cosine similarity...")
    item_sim = compute_topk_cosine_similarity(X_norm, k=50)

    print("Fitting feature-based recommender...")
    knn = ItemKNNRecommender(item_sim)
    knn.fit(ds.train_df)

    
    print("Fitting popularity baseline...")
    pop = PopularityModel()
    pop.fit(ds.train_df)

    eval_kwargs = dict(
        train_df=ds.train_df,
        eval_df=ds.test_df,
        k=10,
        relevance_threshold=4.0,
        item_popularity=ds.item_popularity,
        n_items=ds.n_items,
    )

    print("\nEvaluating models...")

    records = []

    records.append({
        "model": "ItemKNN (genres)",
        **evaluate_ranking_model(knn, **eval_kwargs),
    })

    records.append({
        "model": "Popularity",
        **evaluate_ranking_model(pop, **eval_kwargs),
    })

    df = pl.DataFrame(records).sort("ndcg@10", descending=True)

    print("\n--- Feature validation results ---")
    print(df.select([
        "model",
        "ndcg@10",
        "recall@10",
        "precision@10",
        "coverage",
        "novelty",
    ]))


if __name__ == "__main__":
    main()