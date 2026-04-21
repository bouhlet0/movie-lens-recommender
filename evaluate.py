import numpy as np
import polars as pl
from math import log2


# Rating prediction
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# Per-user ranking metrics
def hit_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    denom = min(k, len(recommended))
    if denom == 0:
        return 0.0
    hits = sum(item in relevant for item in recommended[:k])
    return hits / denom


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(item in relevant for item in recommended[:k])
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_length = min(k, len(relevant))
    idcg = sum(1.0 / log2(rank + 2) for rank in range(ideal_length))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for rank, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (rank + 1)
    return score / min(k, len(relevant))


def mrr_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    for rank, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (rank + 1)
    return 0.0


# Beyond-accuracy (system-level)
def coverage(all_recommendations: list[list[int]], n_items: int) -> float:
    if n_items == 0:
        return 0.0
    unique_recommended = {item for recs in all_recommendations for item in recs}
    return len(unique_recommended) / n_items


def novelty(
    all_recommendations: list[list[int]],
    item_popularity: dict[int, int],
    total_interactions: int,
) -> float:
    if total_interactions == 0:
        return 0.0
    scores = []
    for recs in all_recommendations:
        for item in recs:
            pop = item_popularity.get(item, 1)
            scores.append(-log2(pop / total_interactions))
    return float(np.mean(scores)) if scores else 0.0


# Orchestrators
def evaluate_rating_model(
    model,
    eval_df: pl.DataFrame,
) -> dict[str, float]:
    y_true = eval_df["rating"].to_numpy()
    y_pred = model.predict(eval_df)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
    }


def evaluate_ranking_model(
    model,
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    item_popularity: dict[int, int] | None = None,
    n_items: int | None = None,
) -> dict[str, float]:

    total_interactions = len(train_df)

    relevant_map: dict[int, set[int]] = {}
    for row in (
        eval_df
        .filter(pl.col("rating") >= relevance_threshold)
        .group_by("user_idx")
        .agg(pl.col("item_idx").alias("relevant"))
        .iter_rows(named=True)
    ):
        relevant_map[row["user_idx"]] = set(row["relevant"])

    metrics: dict[str, list[float]] = {
        "precision": [], "recall": [], "ndcg": [],
        "hit":       [], "ap":     [], "mrr":  [],
    }
    all_recommendations: list[list[int]] = []

    train_items_per_user = {
        row["user_idx"]: set(row["item_idx"])
        for row in (
            train_df
            .group_by("user_idx")
            .agg(pl.col("item_idx"))
            .iter_rows(named=True)
        )
    }

    for user_idx, relevant in relevant_map.items():
        if not relevant:
            continue

        train_seen = train_items_per_user.get(user_idx, set())
        seen: set[int] = set()
        recommended: list[int] = []
        for item in model.recommend(user_idx, k):
            if item not in seen and item not in train_seen:
                seen.add(item)
                recommended.append(item)
            if len(recommended) == k:
                break

        all_recommendations.append(recommended)

        metrics["precision"].append(precision_at_k(recommended, relevant, k))
        metrics["recall"].append(recall_at_k(recommended, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(recommended, relevant, k))
        metrics["hit"].append(hit_at_k(recommended, relevant, k))
        metrics["ap"].append(ap_at_k(recommended, relevant, k))
        metrics["mrr"].append(mrr_at_k(recommended, relevant, k))

    results: dict[str, float] = {
        f"precision@{k}":   float(np.mean(metrics["precision"])),
        f"recall@{k}":      float(np.mean(metrics["recall"])),
        f"ndcg@{k}":        float(np.mean(metrics["ndcg"])),
        f"hit_rate@{k}":    float(np.mean(metrics["hit"])),
        f"map@{k}":         float(np.mean(metrics["ap"])),
        f"mrr@{k}":         float(np.mean(metrics["mrr"])),
        "n_eval_users":     float(len(relevant_map)),
        "eval_coverage":    float(len(relevant_map)) / eval_df["user_idx"].n_unique(),
    }

    if n_items is not None:
        results["coverage"] = coverage(all_recommendations, n_items)

    if item_popularity is not None:
        results["novelty"] = novelty(
            all_recommendations,
            item_popularity,
            total_interactions,
        )

    return results

if __name__ == "__main__":
    # Minimal mock model
    class MockModel:
        def __init__(self, recs: list[int]):
            self._recs = recs
        def recommend(self, user_idx: int, k: int) -> list[int]:
            return self._recs

    # Perfect recommender
    assert ndcg_at_k([0, 1, 2], {0, 1, 2}, k=3) == 1.0
    assert precision_at_k([0, 1, 2], {0, 1, 2}, k=3) == 1.0
    assert recall_at_k([0, 1, 2], {0, 1, 2}, k=3) == 1.0
    assert mrr_at_k([0, 1, 2], {0, 1, 2}, k=3) == 1.0
    assert hit_at_k([0, 1, 2], {0, 1, 2}, k=3) == 1.0
    print("OK: perfect recommender")

    # Worst recommender
    assert ndcg_at_k([3, 4, 5], {0, 1, 2}, k=3) == 0.0
    assert mrr_at_k([3, 4, 5], {0, 1, 2}, k=3) == 0.0
    print("OK: worst recommender")

    # Known partial: 1 hit at rank 2 (0-indexed rank 1), k=3
    assert mrr_at_k([3, 0, 4], {0}, k=3) == 0.5
    expected_ndcg = (1.0 / log2(3)) / (1.0 / log2(2))
    assert abs(ndcg_at_k([3, 0, 4], {0}, k=3) - expected_ndcg) < 1e-9
    print("OK: partial hit at rank 2")

    train_df = pl.DataFrame({
        "user_idx": [0, 0, 1, 1, 2, 2],
        "item_idx": [10, 11, 10, 12, 11, 13],
        "rating":   [5.0, 3.0, 4.0, 2.0, 5.0, 4.0],
    })
    # User 2 has no ratings >= 4.0 in eval, so eval_coverage = 2/3
    eval_df = pl.DataFrame({
        "user_idx": [0, 1, 2],
        "item_idx": [5, 6, 7],
        "rating":   [5.0, 4.0, 2.0],
    })
    results = evaluate_ranking_model(
        MockModel([5, 6, 7, 8, 9]),
        train_df, eval_df, k=3, relevance_threshold=4.0
    )
    assert abs(results["eval_coverage"] - 2/3) < 1e-9
    print("OK: eval_coverage")

    train_df2 = pl.DataFrame({
        "user_idx": [0, 0],
        "item_idx": [0, 1],
        "rating":   [5.0, 4.0],
    })
    eval_df2 = pl.DataFrame({
        "user_idx": [0],
        "item_idx": [2],
        "rating":   [5.0],
    })
    # Model returns train items first, then valid items
    results2 = evaluate_ranking_model(
        MockModel([0, 1, 2, 3, 4]),  # 0 and 1 should be excluded
        train_df2, eval_df2, k=3, relevance_threshold=4.0
    )
    assert results2["hit_rate@3"] == 1.0
    print("OK: train item exclusion")

    # k enforcement: model returns 100 items, recommended must be capped at k
    class CountingModel:
        def recommend(self, user_idx, k):
            return list(range(100))  # way more than k
    results3 = evaluate_ranking_model(
        CountingModel(), train_df2, eval_df2, k=3, relevance_threshold=4.0
    )
    print("OK: k enforcement (no crash on oversized model output)")

    print("\nAll checks passed.")