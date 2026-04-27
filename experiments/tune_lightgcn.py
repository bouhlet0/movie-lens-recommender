import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import optuna
from data import build_dataset
from evaluate import evaluate_ranking_model


def objective(trial: optuna.Trial, ds) -> float:
    embedding_dim = trial.suggest_int("embedding_dim",  32,  128, step=32)
    n_layers      = trial.suggest_int("n_layers",        2,    4)
    lr            = trial.suggest_float("lr",          1e-3, 1e-2, log=True)
    reg_weight    = trial.suggest_float("reg_weight",  1e-5, 1e-3, log=True)
    n_epochs      = trial.suggest_int("n_epochs",       20,   60, step=20)

    from models.lightgcn import LightGCNModel
    model = LightGCNModel(
        n_users=ds.n_users,
        n_items=ds.n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        lr=lr,
        reg_weight=reg_weight,
        n_epochs=n_epochs,
        batch_size=4096,
    )
    model.fit(ds.train_df)

    eval_sample = ds.val_df.sample(fraction=0.05, seed=42)
    results = evaluate_ranking_model(
        model=model,
        train_df=ds.train_df,
        eval_df=eval_sample,
        k=10,
        relevance_threshold=4.0,
        item_popularity=ds.item_popularity,
        n_items=ds.n_items,
    )
    return results["ndcg@10"]


def main() -> None:
    print("Building LLN dataset...")
    ds = build_dataset(split="leave_last_n", k=10, lln_n=2)
    print(f"  Users: {ds.n_users:,}  Items: {ds.n_items:,}  "
          f"Train: {ds.n_train:,}  Val: {ds.n_val:,}")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    db_path = (results_dir / "tuning.db").resolve()

    study = optuna.create_study(
        study_name="lightgcn_tuning",
        direction="maximize",
        storage=f"sqlite:///{db_path.as_posix()}",
        load_if_exists=True,
    )

    n_trials = 20
    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    if remaining == 0:
        print(f"Study already has {completed} completed trials. Nothing to run.")
    else:
        print(f"Running {remaining} trials ({completed} already complete)...")
        study.optimize(
            lambda trial: objective(trial, ds),
            n_trials=remaining,
            show_progress_bar=True,
        )

    best = study.best_trial
    print(f"\nBest NDCG@10: {best.value:.4f}")
    print(f"Best params:  {best.params}")

    output = {
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
    }
    out_path = results_dir / "lightgcn_best.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()