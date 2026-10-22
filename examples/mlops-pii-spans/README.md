# A personal-data marker, from documents to a champion

Twelve made-up documents in `docs/` (Montenegrin invoices, a bank statement,
a lease and a payroll list; Turkish invoices, an account summary, a card
statement, a contract and a receipt) and the whole line that turns documents
like them into a model the PII Mask node can use. The line runs under
`deny egress: remote`, so no document leaves the machine:

1. **Data.** `import` reads `docs/` into one JSONL line per document, and
   `prelabel` marks every identifier a checksum or a shape can prove.
2. **Label.** `push` sends the documents and those suggestions to Label Studio
   on 127.0.0.1:8095, a person marks the names, firms and addresses, and
   `pull` brings back what the person confirmed.
3. **Train.** `synthesize` makes 400 made-up documents for training only,
   `split` keeps whole documents together, `dataset-check` refuses a test part
   that is missing a label, and `train` runs `src/train_ner.py`
   (`distilbert-base-multilingual-cased`, eight epochs) with the workspace's
   Python.
4. **Gate, package, gate again, ship.** `span-recall` scores the model's test
   predictions with the checksum rules beside it. `export` writes an int8 ONNX
   model about a quarter of the size, and `span-recall-onnx` scores that model
   again against the deployed champion. `deploy` asks you, then writes
   `models/pii-ner/v<N>/` and `champion.json`.

Every identifier in the documents is generated with a valid checksum and
belongs to nobody; the names, firms and addresses are random pairs of common
ones.

## Set up

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Open the folder in IVYX Studio, choose `.venv` as the workspace's Python,
and run *Label Studio: Start* from the command palette (it needs Docker). The
first training run downloads the base model, about 540 MB, from Hugging Face.

**No Docker?** Copy `labels/gold.example.jsonl` to `data/pii/gold.jsonl`
(the twelve documents labelled for this example) and delete the `push` and
`pull` cards in the MLOps designer.

## Try

| Do this | What you should see |
|---|---|
| Run the pipeline. | It stops at `pull` because nobody has labelled yet, and NEXT STEP offers *Open Label Studio*. The rules have already marked 57 identifiers as suggestions. |
| Label the names, firms and addresses in Label Studio, submit all twelve documents, then run again. | `dataset-check` passes with 412 documents, 12 of them confirmed by you. About two minutes of training on a laptop GPU, then `span-recall` most likely stops the run. On our run it said *ORG recall is 80.0% (4 of 5), under its floor of 98.0%*, because the model cut a firm short before its final `d.o.o.`. Every identifier is still found, because the rules stand beside the model. |
| On this example only, add `thresholds: { ORG: 0.8 }` to both span-recall cards and run again. | The export reports int8 at about a quarter of the size, tagging every token as the model does. The second gate passes, the Deploy card asks you, and `models/pii-ner/v1/` with `champion.json` appears. |
| Run once more. | The model is trained again, and `span-recall-onnx` finds v1 by itself (`champion: auto`) and compares the two on the same documents. A new model that finds less of any label than v1 stops before the deploy. One that is as good asks to become v2. |

The second row is the point of the example. Twelve documents are enough to
run the line and too few to trust a name finder. A real project keeps the
dataset check's default of 200 documents confirmed by a person and the
default recall floors.

Needs `extension-mlops` 1.32.0 or newer, `extension-mlops-nlp` 0.5.0 or
newer, the Secret Manager (it keeps the Label Studio password) and Docker.

## Train on another machine

Only `train` moves; the export and both gates stay here. Add
`extension-compute` and `extension-compute-ssh`, then choose one of two ways
to give the other machine torch and transformers:

- **An image.** On that machine, from a copy of this folder, run
  `docker build -t ivyx-ner compute/`. `compute/Dockerfile` starts from
  PyTorch's CUDA image, adds transformers and bakes the base model in, so a job
  downloads nothing. A machine with no GPU builds it with
  `--build-arg BASE=python:3.12-slim` and gets torch for the CPU.
- **A Python environment** on that machine with torch and transformers
  installed, named by its full path.

`compute/compute.example.yaml` has one target for each. Copy it to
`.punica/compute.yaml` and change the host, or enter the same values in the
Compute panel's form. Then replace the `train` card with this one:

```yaml
  - id: train
    kind: step
    capability: mlops.compute.submit
    input:
      target: gpu-box
      command: python src/train_ner.py --train data/pii/split/train.jsonl --val data/pii/split/val.jsonl --test data/pii/split/test.jsonl --out model
      inputs: [src/train_ner.py, data/pii/split/train.jsonl, data/pii/split/val.jsonl, data/pii/split/test.jsonl]
      outputs: [model/]
      name: pii-ner
    onFailure: stop
```

The job's files go to the other machine over your own ssh, `model/` comes
back, and `span-recall` scores it here as before. A second run with the same
inputs attaches to the job that is already running instead of starting
again.
