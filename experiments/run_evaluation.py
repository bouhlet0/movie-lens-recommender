import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import polars as pl
from data import build_dataset
from evaluate import evaluate_ranking_model
from models.als import ALSModel
from models.mf_svd import MFSVDModel
from models.popularity import PopularityModel


def load_best(name: str, results_dir: Path) -> dict:
    path = results_dir / f"{name}_best.json"
    if not path.exists():
        raise FileNotFoundError(f"No tuning results found at {path}. "
                                f"Run tune_{name}.py first.")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    print("Building LLN dataset...")
    ds = build_dataset(split="leave_last_n", k=10, lln_n=2)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    eval_kwargs = dict(
        train_df=ds.train_df,
        eval_df=ds.test_df,          # test split for final evaluation
        k=10,
        relevance_threshold=4.0,
        item_popularity=ds.item_popularity,
        n_items=ds.n_items,
    )

    records = []

    # Popularity
    print("Fitting Popularity...")
    pop = PopularityModel()
    pop.fit(ds.train_df)
    records.append({"model": "Popularity",
                    **evaluate_ranking_model(model=pop, **eval_kwargs)})

    # Tuned MFSVD
    print("Fitting tuned MFSVD...")
    mfsvd_best = load_best("mfsvd", results_dir)
    mf = MFSVDModel(
        n_users=ds.n_users,
        n_items=ds.n_items,
        **mfsvd_best["best_params"],
    )
    mf.fit(ds.train_df)
    records.append({"model": "MFSVD (tuned)",
                    **evaluate_ranking_model(model=mf, **eval_kwargs)})

    # Tuned ALS
    print("Fitting tuned ALS...")
    als_best = load_best("als", results_dir)
    als = ALSModel(**als_best["best_params"])
    als.fit(ds.train_df, ds.implicit_matrix)
    records.append({"model": "ALS (tuned)",
                    **evaluate_ranking_model(model=als, **eval_kwargs)})

    # Save
    metrics_df = pl.DataFrame(records).sort("ndcg@10", descending=True)
    display_cols = [
        "model", "ndcg@10", "recall@10", "precision@10",
        "hit_rate@10", "map@10", "mrr@10", "coverage", "novelty",
    ]
    print("\n--- Final test set results ---")
    print(metrics_df.select(display_cols))

    out_path = results_dir / "metrics.parquet"
    metrics_df.write_parquet(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()