"""Fill the project's MLflow server with runs to compare and a model to promote.

Nine training runs of `src/train.py`'s baseline over a small grid, logged to
the tracking server `.punica/experiments.yaml` names, then two of them
registered as versions of `titanic-rf`, the first carrying the `champion`
alias. That is the state the experiment-comparison and
model-promotion-review skills are measured against.

    python scripts/seed_mlflow.py            # uses the tracking URI from .punica/experiments.yaml
    python scripts/seed_mlflow.py --reset    # move the champion alias back to version 1

The server has to be up: open the project in IVYX Studio (the Experiments
panel starts it), or run `mlflow server --port 20729 --backend-store-uri
sqlite:///.punica/mlflow.db` from the project root.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

GRID = [
    (100, None, 7),
    (100, 4, 7),
    (100, 8, 7),
    (300, None, 7),
    (300, 4, 7),
    (300, 8, 7),
    (300, None, 11),
    (300, 4, 11),
    (300, 8, 11),
]
MODEL = "titanic-rf"
EXPERIMENT = "titanic-survived"
DATA = "data/titanic/train.csv"
TARGET = "Survived"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--reset", action="store_true", help="only move the champion alias back to version 1")
    args = parser.parse_args(argv)

    import mlflow
    from mlflow import MlflowClient

    import train  # src/train.py

    uri = args.tracking_uri or train.tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    if args.reset:
        client.set_registered_model_alias(MODEL, "champion", "1")
        print(f"{MODEL}: champion -> version 1")
        return 0

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    mlflow.set_experiment(EXPERIMENT)
    frame = pd.read_csv(DATA)
    X, y = train.build_features(frame, TARGET)

    runs = []
    for n_estimators, max_depth, seed in GRID:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        estimator = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        model = train.encoded(estimator, X_train)
        name = f"rf-{n_estimators}-{max_depth}"
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(
                {
                    "n_estimators": n_estimators,
                    "max_depth": str(max_depth),
                    "seed": seed,
                    "data": DATA,
                    "target": TARGET,
                }
            )
            model.fit(X_train, y_train)
            scored = train.metrics("classification", model, X_test, y_test)
            mlflow.log_metrics(scored)
            from mlflow.models import infer_signature

            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=infer_signature(X_train, model.predict(X_train)),
                input_example=X_train.head(5),
                **train.trusted_types_for(model),
            )
            runs.append((run.info.run_id, name, scored["accuracy"]))
            print(f"{name}: accuracy={scored['accuracy']:.4f} run_id={run.info.run_id}")

    # Version 1 is the first run (the one the project shipped with), version 2
    # the best of the rest: a candidate with a training record to compare.
    first = runs[0]
    best = max(runs[1:], key=lambda r: r[2])
    for run_id, _, _ in (first, best):
        mlflow.register_model(f"runs:/{run_id}/model", MODEL)
    client.set_registered_model_alias(MODEL, "champion", "1")
    print(f"{MODEL}: version 1 = {first[1]} ({first[2]:.4f}, champion), version 2 = {best[1]} ({best[2]:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
