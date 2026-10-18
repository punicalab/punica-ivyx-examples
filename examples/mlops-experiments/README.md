# Runs to compare, a model to promote

The Titanic training split, a DVC stage that runs `src/train.py` (the
training script IVYX Studio scaffolds, with MLflow tracking, lineage and a
signed model), and `.punica/experiments.yaml` binding the project to the
MLflow server Studio manages on `localhost:20729`.

## Set up

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Open the folder in IVYX Studio and let the EXPERIMENTS panel start the
server (or start it yourself from the project root:
`mlflow server --port 20729 --backend-store-uri sqlite:///.punica/mlflow.db`).
Then seed it:

```sh
.venv/bin/python scripts/seed_mlflow.py
```

That logs nine random-forest runs over a small grid and registers two of
them as `titanic-rf` version 1 (alias `champion`) and version 2.
`scripts/seed_mlflow.py --reset` moves the alias back to version 1 after a
promotion.

## Try

| Skill | Type into the assistant | A good answer |
|---|---|---|
| `experiment-comparison` | *Which of my MLflow runs is best, and which parameter moved accuracy?* | Lists the runs, then measures the parameter shares with the importance capability instead of guessing; a parameter with a distinct value in every run is reported as excluded, not as important. |
| `model-promotion-review` | *Compare the new version of titanic-rf with the champion and promote it if it is better.* | Reads both versions' training runs through the registry, puts the metrics side by side, recommends, and asks before moving the alias; promotes only after you say yes, through the one governed write. |

Needs `extension-experiments`, `extension-model-registry`, `extension-mlops`
and `extension-evidence`.
