# IVYX examples

Ready-made projects for [IVYX Studio](https://ivyx.io), the desktop
workbench where an AI agent works inside your project under your policy:
every call it makes goes through a gateway that can hold it for your yes,
and every call is on an audit trail you can read back.

## Why this repository exists

**To give a skill something to work on.** The Ivy Marketplace ships
*skills*: procedures an agent loads when your request matches, such as
*look at a new data file before judging it*, *find out why the pipeline
stopped*, or *test my assistant against jailbreak packs*. A skill is only
useful inside a project that has the thing it works on, and a fresh
install has none. Each folder under `examples/` is such a project, small
enough to read in a minute, with a README that says which skills to
install, what to type into the assistant and what a good answer looks
like.

**To show what the measurements were made on.** Before a catalogue skill
is published it is measured: the same request is run with the skill
absent and present, on a local model and on a remote one, and the record
of what the agent did (which tools it called, which files it wrote, what
it answered) decides whether the skill ships. Those records cite the
projects they ran on; `fixtures/` holds those projects exactly as they
were measured, so anyone can open one and run the same request.

**To have a place to put the next one.** A new skill needs a new fixture,
and a fixture that lives on one machine is lost with it. `tools/scrub.sh`
copies a project in with its per-machine state left behind, and
`tools/check.sh` refuses a tree that still carries a path or a secret.

## How to use it

1. Install IVYX Studio and open a folder from `examples/` with
   *File › Open Folder* (or run `ivyx` inside it from a terminal).
2. Install the skills the folder's README names from the Ivy Marketplace's
   SKILLS tab, and any extension it says it needs.
3. Type the README's request into the assistant. The skill is not chosen
   by hand: a request in those words reaches it on its own, and the
   README says what the answer should contain.

| Folder | Who it is for | Skills to try | Needs |
|---|---|---|---|
| `examples/data-first-look` | a data scientist opening a new file | `dataset-first-look`, `train-test-leakage-check` | extension-duckdb, extension-profiler |
| `examples/mlops-pipeline` | an ML engineer with a pipeline and a data contract | `pipeline-run-triage`, `data-contract-authoring`, `notebook-to-pipeline-stage` | extension-mlops, -mlops-dvc, -mlops-pandera, -test-pytest, -ipynb-viewer; a Python env from `requirements.txt` |
| `examples/mlops-experiments` | a data scientist with runs in MLflow and a model to promote | `experiment-comparison`, `model-promotion-review` | extension-experiments, -model-registry, -mlops, -evidence; a Python env, then `scripts/seed_mlflow.py` |
| `examples/mlops-local-champion` | an ML engineer whose policy keeps data on the machine | `pipeline-run-triage` (and the pipeline itself: a gate, a deploy, a champion that stops a weaker model) | extension-mlops 1.32+, -mlops-dvc, -experiments 2.29+; a Python env from `requirements.txt` |
| `examples/mlops-pii-spans` | a team training a model that finds personal data in documents | the pipeline itself: label, train, gate, int8 export, deploy | extension-mlops 1.32+, -mlops-nlp 0.5+, -secret-manager; Docker for Label Studio; a Python env with torch |
| `examples/llmops-support` | an LLM developer with a support assistant over policy documents | `rag-knowledge-base-setup`, `eval-suite-authoring`, `guardrails-pack-check` | extension-knowledge, -vectors, -evals, -evals-text, -guardrails, -llms-instruction-class-service; an AI core |
| `examples/student-notebooks` | a student with a notebook and a homework | `concept-practice`, `notebook-error-hint`, `assignment-starter` | extension-ipynb-viewer, -duckdb, a Python kernel |

## What is in here

```
examples/     seven projects to open and try, one per audience, each with a README
fixtures/     the fourteen projects the catalogue skills were measured on, as measured
              (fixtures/README.md maps each to its measurement section and says how to run one again)
tools/        scrub.sh — copy a project in without its machine state; check.sh — refuse paths and secrets
DATA.md       where the data comes from (Titanic is public domain; everything else is synthetic)
LICENSE       MIT
```

The measurement record the fixtures belong to is
`docs/catalog-measurements.md` in the `extension-skills` repository, and the
method for writing and measuring a catalogue skill is `catalog/README.md`
there.

## Adding a project

Copy it in through the scrubber from the repository root, write its
README in the shape of the existing ones (what it is, how to set it up,
a *Try* table of skill → request → good answer, what it needs), and run
the check before committing:

```sh
sh tools/scrub.sh /path/to/project examples/<name>
sh tools/check.sh
```

A fixture for a measurement goes under `fixtures/` the same way, with
`keep-events` as the third argument when the skill under test reads the
audit trail.
