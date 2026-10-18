"""Train a model and record the run, scaffolded by IVYX Studio.

**What is yours and what is ours.** The model choice, the features and the
hyperparameters are your work and this file leaves them to you: they live in
`build_features()` and `build_model()` below, both marked. Everything around
them is plumbing that breaks CI when each project reinvents it, so it is written
for you: the tracking wiring, the editor trace tag, the dataset lineage record,
the artifact and signature, the run id on stdout, and the exit codes.

Run it directly or from a pipeline stage:

    python src/train.py
    python src/train.py --data data/prepared.csv --experiment churn

The gate that judges this run (`mlops.model.evaluate`) reads metrics from the
tracking server, not from this script's output, so anything you want gated has
to be logged here or in `evaluate.py` — a metric that is only printed cannot be
checked by anything.
"""

import argparse
import os
import sys
import traceback

# ---- configuration, rewritten when this file is scaffolded ----
EXPERIMENT = "titanic-survived"
DATA = "data/titanic/train.csv"
TARGET = "Survived"
MODEL_NAME = ""
TASK = "classification"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# A text column with more distinct values than this is a name, a ticket or a
# free text, not a category, and is left out of the baseline.
MAX_CATEGORIES = 20


def trusted_types_for(model):
    """Ours. What a current MLflow needs to save the model this script built.

    MLflow 3.x serialises a scikit-learn model with skops, which refuses to
    write a tree-based estimator unless its node storage
    (`sklearn.tree._tree.Tree`) is declared trusted: `log_model` raises
    "The saved sklearn model references untrusted types" and the run has no
    model. That warning is about files from somewhere else. This model was
    fitted a few lines above in this process, which is the one case MLflow's
    own message allows, so the types skops finds in it are passed as trusted
    and the saved flavor carries them for whoever loads the model. An older
    MLflow has no such parameter and is given nothing. Measured 2026-10-08:
    3.17.0 refused the baseline RandomForest, 2.22.5 saved it as before.
    """
    import inspect

    import mlflow
    import mlflow.sklearn

    try:
        params = inspect.signature(mlflow.sklearn.log_model).parameters
    except (TypeError, ValueError):
        return {}
    if "skops_trusted_types" not in params:
        return {}
    try:
        import io

        from skops.io import dump, get_untrusted_types

        buffer = io.BytesIO()
        dump(model, buffer)
        return {"skops_trusted_types": get_untrusted_types(data=buffer.getvalue())}
    except Exception:
        # No skops, or a model skops cannot write: the pickle format every
        # MLflow before 3.x used is still accepted.
        return {"serialization_format": "cloudpickle"}


def build_features(frame, target):
    """Yours. Return `(X, y)`.

    The scaffold drops the target column and hands the rest to `encoded()`,
    which keeps the numbers and the low-cardinality categories. That is a
    starting point rather than a feature set: selection, scaling and anything
    the data means belong here, and the model is only as good as what this
    returns.
    """
    y = frame[target]
    X = frame.drop(columns=[target])
    return X, y


def feature_columns(X):
    """Ours. Which columns the baseline learns from, by what they hold.

    A number is kept unless it is a row id (an integer that never repeats).
    Anything else is a category when it has few distinct values and is left
    out when it has many. Before 2.27.0 only the numbers were kept, so a table
    whose signal sits in a category trained without it: on the Titanic table
    the sex column was dropped and the passenger id was kept.
    """
    import pandas as pd

    numeric, categorical = [], []
    for column in X.columns:
        values = X[column]
        distinct = values.nunique(dropna=True)
        if pd.api.types.is_bool_dtype(values):
            categorical.append(column)
        elif pd.api.types.is_numeric_dtype(values):
            row_id = (
                pd.api.types.is_integer_dtype(values)
                and len(values) > MAX_CATEGORIES
                and distinct == len(values)
            )
            if not row_id:
                numeric.append(column)
        elif 0 < distinct <= MAX_CATEGORIES:
            categorical.append(column)
    return numeric, categorical


def encoded(estimator, X):
    """Ours. The estimator behind the encoding it needs.

    The encoding is part of the logged model rather than a step before it, so
    whatever loads the model (`evaluate.py`, a server) hands it the table as it
    is and gets the same columns the training saw.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric, categorical = feature_columns(X)
    if not numeric and not categorical:
        raise SystemExit(
            "No usable column: every column is a row id or free text. "
            "Write build_features() for this table."
        )
    steps = []
    if numeric:
        steps.append(("numeric", "passthrough", numeric))
    if categorical:
        steps.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(steps, remainder="drop")),
            ("model", estimator),
        ]
    )


def build_model(task, random_state):
    """Yours. Return an unfitted estimator.

    A deliberately unremarkable baseline: the point of a scaffold is that the
    first run works, not that it is good. Swap in the model you mean to use, and
    log its hyperparameters through `params()` so a run can be reproduced from
    the record alone.
    """
    if task == "regression":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=200, random_state=random_state)
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=200, random_state=random_state)


def params(model, args):
    """What the run records about how it was produced."""
    values = {
        "task": args.task,
        "data": args.data,
        "target": args.target,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    try:
        values.update({k: v for k, v in model.get_params().items() if v is not None})
    except Exception:
        pass
    return values


def metrics(task, model, X_test, y_test):
    """Ours. The generic half of scoring, so the numbers mean the same thing
    in every project and the gate can compare a run with a champion.

    Slice metrics are `evaluate.py`'s job — this is the overall picture only.
    """
    import numpy as np

    predictions = model.predict(X_test)
    if task == "regression":
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    average = "binary" if len(set(y_test)) <= 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, average=average, zero_division=0)),
        "precision": float(
            precision_score(y_test, predictions, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, average=average, zero_division=0)
        ),
    }



def tracking_uri():
    """Where this run is recorded, in the order that keeps CI and the editor honest.

    1. `MLFLOW_TRACKING_URI` from the environment. CI sets it as a repository
       secret and it has to win: a runner cannot reach a tracking server running
       on your laptop.
    2. The workspace's configured backend, read from `.punica/experiments.yaml`.
       This is the server the editor's Experiments panel manages. Without this
       step a run started by a pipeline stage lands in a local `./mlruns`
       directory that nothing else reads, and the model quality gate then reports
       that it has no run to judge, which is true and useless.
    3. Nothing, and MLflow falls back to its own default.

    PyYAML is not an extra dependency here; MLflow requires it.
    """
    from_env = os.environ.get("MLFLOW_TRACKING_URI")
    if from_env:
        return from_env

    # Walk up, because a stage may be run from a subdirectory.
    here = os.path.abspath(os.getcwd())
    for _ in range(5):
        candidate = os.path.join(here, ".punica", "experiments.yaml")
        if os.path.isfile(candidate):
            try:
                import yaml

                with open(candidate) as handle:
                    config = yaml.safe_load(handle) or {}
                wanted = config.get("defaultBackendId")
                for backend in config.get("backends") or []:
                    if wanted and backend.get("id") != wanted:
                        continue
                    uri = (backend.get("config") or {}).get("trackingUri")
                    if uri:
                        return uri
            except Exception as exc:
                # A workspace file we cannot read is not a reason to fail a
                # training run; MLflow's default still records something. But
                # it must not be silent: the docstring above describes exactly
                # what goes wrong next, and the commonest cause is a Python
                # without PyYAML, where the run lands in a local ./mlruns that
                # the quality gate never looks in.
                print(
                    f"warning: could not read {candidate} ({exc}), so this run is"
                    " recorded wherever MLflow defaults to rather than the"
                    " workspace's tracking server.",
                    file=sys.stderr,
                )
                return None
            return None
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None

def correlation_id():
    """The editor trace this run belongs to, or `None`.

    Set by whatever spawned this process: IVYX Studio passes
    `PUNICA_CORRELATION_ID` down into a DVC stage, and nothing sets it when you
    run this file by hand or when CI runs it. Absent is the honest answer there.
    A trace invented here would let an evidence package claim this model was
    produced by calls that were never made.
    """
    value = (os.environ.get("PUNICA_CORRELATION_ID") or "").strip()
    return value or None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a model and record the run.")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--task", default=TASK, choices=["classification", "regression"])
    parser.add_argument("--model-name", default=MODEL_NAME, dest="model_name")
    parser.add_argument("--test-size", default=TEST_SIZE, type=float, dest="test_size")
    parser.add_argument(
        "--random-state", default=RANDOM_STATE, type=int, dest="random_state"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import mlflow
    import pandas as pd
    from sklearn.model_selection import train_test_split

    uri = tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(args.experiment)

    frame = pd.read_csv(args.data)
    if args.target not in frame.columns:
        raise SystemExit(
            f"{args.data} has no column '{args.target}'. Pass --target, or edit TARGET."
        )

    X, y = build_features(frame, args.target)
    stratify = y if args.task == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=stratify
    )

    estimator = build_model(args.task, args.random_state)
    model = encoded(estimator, X_train)

    with mlflow.start_run() as run:
        trace = correlation_id()
        if trace:
            # The one bridge between two id spaces. `experiments.log` writes the
            # same tag for a run an agent logged; this writes it for a run the
            # editor's pipeline trained, so both are reachable from a model
            # version by the same walk.
            mlflow.set_tag("punica.correlationId", trace)

        # Lineage is free and almost nobody records it: this pins the run to the
        # dataset's content digest, so "which data produced this model" is
        # answerable a year later without trusting a filename.
        try:
            mlflow.log_input(
                mlflow.data.from_pandas(frame, source=args.data, name=args.data),
                context="training",
            )
        except Exception:
            # An older tracking server without the datasets API must not cost a
            # training run.
            pass

        mlflow.log_params(params(estimator, args))
        numeric, categorical = feature_columns(X_train)
        mlflow.log_params(
            {"features_numeric": len(numeric), "features_categorical": len(categorical)}
        )
        model.fit(X_train, y_train)

        scored = metrics(args.task, model, X_test, y_test)
        mlflow.log_metrics(scored)

        from mlflow.models import infer_signature

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=args.model_name or None,
            **trusted_types_for(model),
        )

        # Printed for a log a person reads. Nothing downstream parses it: the
        # gate resolves the run through the tracking server, which stays true
        # even when a stage is retried or run out of order.
        print(f"run_id={run.info.run_id}")
        for name, value in sorted(scored.items()):
            print(f"{name}={value:.6f}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        # A traceback on stderr, a clean non-zero exit, and nothing half-logged
        # pretending the run succeeded.
        traceback.print_exc()
        sys.exit(1)

# retrain after the tracking server came up
