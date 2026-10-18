# Measurement fixtures

The projects the catalogue skills were measured on, copied from the
measuring machine through `tools/scrub.sh` (no trail, no search index, no
Python environment record, no caches; the settings file without its
per-machine `system` block). `audit2` keeps its audit trail, with the
machine's paths replaced by `<workspace>`, `<machine>` and `<home>`, because
the skill it measures reads nothing else (its `.punica/.gitignore` ends with
`!/events/` for that reason; every other project's allowlist leaves the
trail out, as Studio writes it).

| Folder | Measurement section in `extension-skills/docs/catalog-measurements.md` | Skill |
|---|---|---|
| `ds1` | dataset-first-look | `dataset-first-look` |
| `ds2` | train-test-leakage-check (ds1 plus `data/test.csv`) | `train-test-leakage-check` |
| `rag1` | rag-knowledge-base-setup | `rag-knowledge-base-setup` |
| `mlops1` | pipeline-run-triage ("the triage project") | `pipeline-run-triage` |
| `contract1` | data-contract-authoring (mlops1 with the contract stage and `contracts/` removed) | `data-contract-authoring` |
| `nb2pipe` | notebook-to-pipeline-stage (mlops1 plus `notebooks/train_v2.ipynb`) | `notebook-to-pipeline-stage` |
| `audit2` | agent-run-audit (mlops1's trail cut at the end of run `77a141e5…`, 39 records) | `agent-run-audit` (parked) |
| `exp1` | experiment-comparison, model-promotion-review (the MLflow store is rebuilt by `examples/mlops-experiments/scripts/seed_mlflow.py`) | `experiment-comparison`, `model-promotion-review` |
| `nb1` | notebook-error-hint (`odev2.ipynb`, Turkish) | `notebook-error-hint` |
| `nb2` | concept-practice (`pratik.ipynb`, Turkish title) | `concept-practice` |
| `nb2en` | concept-practice, the English notebook it was published on | `concept-practice` |
| `nb3` | assignment-starter (`odev.md`, `odev3.ipynb`, Turkish) | `assignment-starter` |
| `evals1` | eval-suite-authoring | `eval-suite-authoring` |
| `guard1` | guardrails-pack-check (evals1 again) | `guardrails-pack-check` |

## Running a measurement again

The driver is `punica-editor/docs/sighting-tools/skill-run.mjs`
(`punicalab/punica-editor`). It drives `runAgent` in a running IVYX Studio
over its Chrome DevTools port, answers every held call `once`, restores the
workspace from the fixture before each run and appends one line per run to
`ledger/<label>.jsonl`.

1. Start Studio with `--remote-debugging-port=9333`.
2. Copy the fixture to a working folder (the driver restores it from the
   fixture, so the fixture itself is never written to) and put the skill
   under test inside it as a project skill: `.punica/skills/<name>/` holding
   its `SKILL.md`. A run never chooses the skill; whether it loads it is the
   measurement.
3. `node skill-run.mjs <label> <working-folder> <fixture> 2 '<request>'`, with
   `CORE_ID=remote:<id>` to pick the core (see the driver's header for the
   current spelling), `RESTORE_EVENTS=1` for `audit2`, and `RUN_MS` above
   the default for a skill whose procedure runs suites.

The requests are quoted in each measurement section.
