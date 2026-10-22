# A gate and a champion that never leave the machine

A churn model trained on 600 synthetic customers, under the one policy rule
a team with personal data writes first: `deny egress: remote` in
`.punica/policy.yaml`, *no data leaves this machine*. That rule refuses the
MLflow gate (`mlops.model.evaluate` declares `net.request`), so this pipeline
gates and ships a model with no tracking server in the path:

1. `train` runs `src/train.py` through DVC. The script is the one IVYX Studio
   scaffolds; beside its MLflow run (kept in a local `mlruns/` folder) it
   writes `out/model/`: the model, its test predictions as `predictions.csv`
   and an `ivyx-model.json` card.
2. `check-predictions` scores that file itself: F1 must be at least 0.5, each
   region within 5 points of the champion's F1 on that region, and the whole
   test part no worse than the champion on the rows both models predicted.
3. `ship` asks you, refuses a gate that did not pass, and copies the model
   into `models/churn/v<N>/` with `models/churn/champion.json` beside it. The
   next run's check finds that champion by itself (`champion: auto`).

## Set up

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`dvc.yaml` runs the script with `.venv/bin/python`, so the environment has to
be at that path. Open the folder in IVYX Studio and open the MLOps designer
from the activity bar: three cards, *Run* beside them.

## Try

| Do this | What you should see |
|---|---|
| Run the pipeline once. | The check passes at F1 54.0% with no champion to compare yet, and the Ship card asks before it writes. After your yes, `models/churn/v1/` holds the model and its predictions, and `champion.json` names v1. |
| In `src/train.py`, make `build_model()` return `LogisticRegression(max_iter=1000, class_weight="balanced")` (import it from `sklearn.linear_model`), then run again. | F1 70.0% against the champion's 54.0% on the same test set; the Ship card asks again and v2 becomes the champion, with `previousVersion: v1`. |
| Put the random forest back and run a third time. | The check stops the run: *f1 is 54.0%, under the champion v2's 70.0% on the same test set.* Ship is skipped and NEXT STEP says why. |
| Ask the assistant *The pipeline stopped. Why, and what do I do?* (`pipeline-run-triage`) | It reads the newest run from the record, names the check and the champion's number, and does not run the pipeline again. |

Run the pipeline twice with nothing changed and DVC skips the training, so
the check scores the same predictions and the deploy reports that the model
is already the champion instead of opening a new version.

## From a terminal

The same run needs no window. In one terminal:

```sh
ivyx flow run mlops --here
```

In a second terminal, `ivyx approve --watch` shows the Ship card when the
check has passed; `a` writes the version and `d` stops the run with nothing
written. The model is trained with `.venv`, so the folder needs that
environment and nothing selected in Studio.

Needs `extension-mlops` 1.32.0 or newer, `extension-mlops-dvc` and
`extension-experiments` 2.29.0 or newer (the training script's model
folder). The terminal run needs `extension-mlops` 1.33.0 and `ivyx` 0.26.0. `pipeline-run-triage` comes from the Ivy Marketplace's SKILLS tab.
