# from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp


def build_genre_vocabulary(movies_df: pl.DataFrame) -> dict[str, int]:
    
    genres = (
        movies_df
        .select(pl.col("genres").str.split("|"))
        .explode("genres")
        .unique()
        .drop_nulls()
        .to_series()
        .to_list()
    )

    return {g: i for i, g in enumerate(sorted(genres))}


def build_item_feature_matrix(
    movies_df: pl.DataFrame,
    item2idx: dict[int, int],
) -> tuple[sp.csr_matrix, dict[str, int]]:

    genre2idx = build_genre_vocabulary(movies_df)

    rows = []
    cols = []
    data = []

    for row in movies_df.iter_rows(named=True):
        movie_id = row["movieId"]
        genres = row["genres"]

        # Skip items not in training vocab (important)
        if movie_id not in item2idx:
            continue

        item_idx = item2idx[movie_id]

        if genres is None or genres == "(no genres listed)":
            continue

        for g in genres.split("|"):
            if g in genre2idx:
                rows.append(item_idx)
                cols.append(genre2idx[g])
                data.append(1.0)

    n_items = len(item2idx)
    n_features = len(genre2idx)

    X = sp.csr_matrix(
        (np.array(data, dtype=np.float32), (rows, cols)),
        shape=(n_items, n_features),
    )

    return X, genre2idx


def normalize_rows(X: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(X.power(2).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    return X.multiply(1.0 / norms[:, None])

def compute_topk_cosine_similarity(X: sp.csr_matrix, k: int = 50) -> sp.csr_matrix:
    X = normalize_rows(X)
    
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0.0)

    n_items = sim.shape[0]
    rows, cols, data = [], [], []

    for i in range(n_items):
        top_idx = np.argpartition(sim[i], -k)[-k:]
        top_idx = top_idx[np.argsort(-sim[i, top_idx])]
        for j in top_idx:
            if sim[i, j] > 0:
                rows.append(i)
                cols.append(j)
                data.append(sim[i, j])

    return sp.csr_matrix(
        (np.array(data, dtype=np.float32), (rows, cols)),
        shape=(n_items, n_items),
    )