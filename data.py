import polars as pl
from pathlib import Path
import scipy.sparse as sp
import numpy as np
from typing import Optional
from dataclasses import dataclass

DATA_DIR = Path("./data/ml-32m/")

def load_raw(data_dir: Path = DATA_DIR) -> tuple[pl.DataFrame, pl.DataFrame]:
    ratings = pl.read_csv(
        data_dir / "ratings.csv",
        schema_overrides={
            "userId":   pl.Int32,
            "movieId":  pl.Int32,
            "rating":   pl.Float32,
            "timestamp": pl.Int64,
        }
    ).with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("datetime")
    )

    movies = pl.read_csv(
        data_dir / "movies.csv",
        schema_overrides={
            "movieId":  pl.Int32,
            "title":    pl.Utf8,
            "genres":   pl.Utf8,
        }
    )
    return ratings, movies

def k_core_filter(ratings: pl.DataFrame, k: int = 10) -> pl.DataFrame:
    while True:
        valid_movies = (
            ratings.group_by("movieId")
                   .agg(pl.len().alias("n"))
                   .filter(pl.col("n") >= k)
                   .select("movieId")
        )
        valid_users = (
            ratings.group_by("userId")
                   .agg(pl.len().alias("n"))
                   .filter(pl.col("n") >= k)
                   .select("userId")
        )
        filtered = ratings.join(valid_movies, on="movieId", how="inner") \
                          .join(valid_users,  on="userId",  how="inner")

        if len(filtered) == len(ratings):
            break
        ratings = filtered

    return filtered

def remap_ids(
    train:  pl.DataFrame,
    val:    pl.DataFrame,
    test:   pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[int, int], dict[int, int]]:
    unique_users = train["userId"].unique().sort()
    unique_items = train["movieId"].unique().sort()

    user_map = pl.DataFrame({
        "userId":   unique_users,
        "user_idx": pl.int_range(len(unique_users), eager=True).cast(pl.Int32),
    })
    item_map = pl.DataFrame({
        "movieId":  unique_items,
        "item_idx": pl.int_range(len(unique_items), eager=True).cast(pl.Int32),
    })

    def apply_remap(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df
            .join(user_map, on="userId",  how="left")
            .join(item_map, on="movieId", how="left")
            .drop_nulls(["user_idx", "item_idx"])
        )

    train = apply_remap(train)
    val = apply_remap(val)
    test = apply_remap(test)

    user2idx = dict(zip(user_map["userId"].to_list(),  user_map["user_idx"].to_list()))
    item2idx = dict(zip(item_map["movieId"].to_list(), item_map["item_idx"].to_list()))

    return train, val, test, user2idx, item2idx

def temporal_split(
    ratings: pl.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    timestamps = ratings["timestamp"]
    val_cutoff = timestamps.quantile(1.0 - val_frac - test_frac)
    test_cutoff = timestamps.quantile(1.0 - test_frac)

    train = ratings.filter(pl.col("timestamp") <  val_cutoff)
    val = ratings.filter((pl.col("timestamp") >= val_cutoff) & (pl.col("timestamp") < test_cutoff))
    test = ratings.filter(pl.col("timestamp") >= test_cutoff)

    return train, val, test

def to_sparse_matrix(
    df: pl.DataFrame,
    n_users: int,
    n_items: int,
) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            df["rating"].to_numpy(),
            (df["user_idx"].to_numpy(), df["item_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )

def compute_item_popularity(train_df: pl.DataFrame) -> dict[int, int]:
    return dict(
        train_df
        .group_by("item_idx")
        .agg(pl.len().alias("count"))
        .iter_rows()
    )

def to_implicit_matrix(
    df: pl.DataFrame,
    n_users: int,
    n_items: int,
    alpha: float = 40.0,
) -> sp.csr_matrix:
    confidence = 1.0 + alpha * df["rating"].to_numpy().astype(np.float32)
    return sp.csr_matrix(
        (
            confidence,
            (df["user_idx"].to_numpy(), df["item_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )
    
def leave_last_n_split(
    ratings: pl.DataFrame,
    n: int = 2,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Per-user leave-last-N split ordered by timestamp.

    - Test:  last N interactions per user
    - Val:   second-to-last N interactions per user
    - Train: everything else

    Users with fewer than (2N + 1) interactions will not appear in val/test.
    """

    if n < 1:
        raise ValueError("n must be >= 1")

    two_n = 2 * n

    ranked = ratings.with_columns(
        pl.col("timestamp")
        .rank(method="ordinal", descending=True)
        .over("userId")
        .alias("rank")
    )

    test_mask  = pl.col("rank") <= n
    val_mask   = (pl.col("rank") > n) & (pl.col("rank") <= two_n)
    train_mask = pl.col("rank") > two_n

    test  = ranked.filter(test_mask).drop("rank")
    val   = ranked.filter(val_mask).drop("rank")
    train = ranked.filter(train_mask).drop("rank")

    return train, val, test
@dataclass
class Dataset:
    train_df:   pl.DataFrame
    val_df:     pl.DataFrame
    test_df:    pl.DataFrame
    movies_df:  pl.DataFrame

    user2idx:   dict[int, int]
    item2idx:   dict[int, int]
    idx2user:   dict[int, int]
    idx2item:   dict[int, int]

    n_users:    int
    n_items:    int

    item_popularity: dict[int, int]
    train_matrix:    sp.csr_matrix
    implicit_matrix: sp.csr_matrix

    # metadata
    n_train:    int
    n_val:      int
    n_test:     int
    sparsity:   float

    # evaluation metadata (LLN-specific, optional)
    n_eval_users: Optional[int] = None
    eval_coverage: Optional[float] = None


def build_dataset(
    data_dir: Path = DATA_DIR,
    k: int = 10,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split: str = "temporal",
    lln_n: int = 2,
) -> Dataset:
    ratings, movies = load_raw(data_dir)
    ratings = k_core_filter(ratings, k=k)

    if split == "temporal":
        train_df, val_df, test_df = temporal_split(ratings, val_frac, test_frac)
        n_eval_users = None
        eval_coverage = None

    elif split == "leave_last_n":
        train_df, val_df, test_df = leave_last_n_split(ratings, n=lln_n)

        n_total_users = ratings["userId"].n_unique()
        n_eval_users = (
            ratings
            .group_by("userId")
            .agg(pl.len().alias("n_interactions"))
            .filter(pl.col("n_interactions") >= 2 * lln_n + 1)
            .height
        )
        eval_coverage = n_eval_users / n_total_users

    else:
        raise ValueError(f"Unknown split: '{split}'")

    train_df, val_df, test_df, user2idx, item2idx = remap_ids(
        train_df, val_df, test_df
    )

    train_users = train_df.select("user_idx").unique()
    train_items = train_df.select("item_idx").unique()

    val_df = (
        val_df
        .join(train_users, on="user_idx", how="inner")
        .join(train_items, on="item_idx", how="inner")
    )

    test_df = (
        test_df
        .join(train_users, on="user_idx", how="inner")
        .join(train_items, on="item_idx", how="inner")
    )

    # --- Dimensions ---
    n_users = len(user2idx)
    n_items = len(item2idx)

    # --- Matrices & stats ---
    train_matrix    = to_sparse_matrix(train_df, n_users, n_items)
    implicit_matrix = to_implicit_matrix(train_df, n_users, n_items)
    item_popularity = compute_item_popularity(train_df)

    sparsity = 1.0 - train_matrix.nnz / (n_users * n_items)

    return Dataset(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        movies_df=movies,

        user2idx=user2idx,
        item2idx=item2idx,
        idx2user={v: k for k, v in user2idx.items()},
        idx2item={v: k for k, v in item2idx.items()},

        n_users=n_users,
        n_items=n_items,

        train_matrix=train_matrix,
        implicit_matrix=implicit_matrix,
        item_popularity=item_popularity,

        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        sparsity=sparsity,

        n_eval_users=n_eval_users,
        eval_coverage=eval_coverage,
    )
    
if __name__ == "__main__":
    import time

    print("Building dataset...")
    t0 = time.time()
    ds = build_dataset()
    elapsed = time.time() - t0

    print("\n--- Split sizes ---")
    print(f" Train: {ds.n_train:>10,} interactions")
    print(f" Val: {ds.n_val:>10,} interactions")
    print(f" Test: {ds.n_test:>10,} interactions")

    print("\n--- Vocabulary ---")
    print(f" Users: {ds.n_users:>10,}")
    print(f" Items: {ds.n_items:>10,}")

    print("\n--- Train matrix ---")
    print(f" Shape: {ds.train_matrix.shape}")
    print(f" Sparsity: {ds.sparsity * 100:.4f}%")
    print(f" nnz: {ds.train_matrix.nnz:,}")

    print("\n--- Cold-start bleed check ---")
    val_users = set(ds.val_df["user_idx"].to_list())
    val_items = set(ds.val_df["item_idx"].to_list())
    test_users = set(ds.test_df["user_idx"].to_list())
    test_items = set(ds.test_df["item_idx"].to_list())
    train_users = set(ds.train_df["user_idx"].to_list())
    train_items = set(ds.train_df["item_idx"].to_list())

    print(f" Val users unseen in train: {len(val_users - train_users)}")
    print(f" Val items unseen in train: {len(val_items - train_items)}")
    print(f" Test users unseen in train: {len(test_users - train_users)}")
    print(f" Test items unseen in train: {len(test_items - train_items)}")

    print("\n--- Timestamp sanity check ---")
    print(f" Train max datetime: {ds.train_df['datetime'].max()}")
    print(f" Val min datetime: {ds.val_df['datetime'].min()}")
    print(f" Val max datetime: {ds.val_df['datetime'].max()}")
    print(f" Test min datetime: {ds.test_df['datetime'].min()}")
    print(f" Test max datetime: {ds.test_df['datetime'].max()}")

    print(f"\nDone in {elapsed:.1f}s")