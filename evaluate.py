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

    for user_idx, relevant in relevant_map.items():
        if not relevant:
            continue

        seen: set[int] = set()
        recommended: list[int] = []
        for item in model.recommend(user_idx, k):
            if item not in seen:
                seen.add(item)
                recommended.append(item)

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