# A pipeline with a data contract

A three-stage MLOps pipeline on DVC: a Pandera data contract over
`data/train.csv`, the pytest suite in `tests/`, then `src/train.py`, which
fits a logistic regression and writes `model.pkl` and `metrics.json`. The
pipeline is declared in `.punica/mlops.yaml`; `.punica/policy.yaml` holds
the house rule that a person says yes before anything writes to disk.

The data has ten rows labelled `2` that the contract (`label ∈ {0, 1}`)
rejects, so a run stops at the first stage. That is the point.

## Set up

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`dvc.yaml` runs the training stage with `.venv/bin/python`, so the
environment has to be at that path. Open the folder in IVYX Studio; the
MLOPS panel shows the three stages and a *Run pipeline* button.

## Try

| Skill | Type into the assistant | A good answer |
|---|---|---|
| `pipeline-run-triage` | *The pipeline failed. Why, and what do I do?* (after one run from the MLOPS panel) | Reads the newest run from the record: the `pandera` stage stopped it, ten rows with `label = 2`, the contract line to look at; it does not run the pipeline again. |
| `data-contract-authoring` | Remove the `pandera` stage from `.punica/mlops.yaml` and the `contracts/` folder, then: *I want a data contract for data/train.csv so the pipeline refuses bad data before training.* | Profiles the file, writes `contracts/train_contract.py` through the drafting capability, adds the validate gate before `train` (held for your yes), and lists the label's three values with `pa.Check.isin([0, 1, 2])` as the line to change. |
| `notebook-to-pipeline-stage` | *Turn notebooks/train_v2.ipynb into a pipeline stage.* | Extracts `src/train_v2.py` and adds a DVC stage named after the notebook; the notebook itself is never a dependency of the stage. |

Needs `extension-mlops`, `extension-mlops-dvc`, `extension-mlops-pandera`,
`extension-test-pytest` and, for the notebook, `extension-ipynb-viewer`.
