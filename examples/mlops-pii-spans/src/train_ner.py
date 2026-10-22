"""
Train a span tagger on the text data steps' JSONL (punica-editor/docs/nlp-data.md).

Written into the workspace by *NLP: Scaffold the Training Script*
(extension-mlops-nlp). The configuration block below is yours: the base model,
the epochs, the learning rate. The rest is the contract the pipeline reads, so
keep it when you change the model:

  - records in, one document per line, span offsets in UTF-16 code units;
  - <out>/ holds the model, its tokenizer and ivyx-ner.json;
  - <out>/predictions.<part>.jsonl holds every document of that part with the
    spans this model found, which mlops.nlp.evaluate scores itself;
  - metrics.json maps flat names to numbers, which the Compute hub logs;
  - stdout carries one JSON line at the end and nothing else, because the task
    runner keeps only the first 4 MB of it; progress goes to <out>/train.log.

It runs the same here (mlops.nlp.train) and on another machine
(mlops.compute.submit). It needs torch and transformers:

    python -m pip install torch transformers
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

# ---- configuration: yours ----
BASE_MODEL = "distilbert-base-multilingual-cased"
EPOCHS = 8
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
MAX_LENGTH = 256
STRIDE = 64
WARMUP = 0.1
SEED = 13
DEVICE = ""  # cuda, mps or cpu; the fastest this machine has when empty
# ---- end of configuration ----


def finite(value):
    """JSON has no NaN: a number that is not finite is left out rather than written as one."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items() if not (isinstance(v, float) and not math.isfinite(v))}
    return value


def answer(obj, code=0):
    sys.stdout.write(json.dumps(finite(obj), ensure_ascii=False) + "\n")
    sys.stdout.flush()
    sys.exit(code)


_missing = [p for p in ("torch", "transformers") if importlib.util.find_spec(p) is None]
if _missing:
    answer({"ok": False, "error": "needs-packages", "packages": _missing}, 3)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()


# ---- the records ----


def utf16_table(text):
    """For each code point index, the UTF-16 index it starts at (len(text) + 1 entries)."""
    table = [0] * (len(text) + 1)
    n = 0
    for i, ch in enumerate(text):
        table[i] = n
        n += 2 if ord(ch) > 0xFFFF else 1
    table[len(text)] = n
    return table


def read_jsonl(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record["text"]
            table = utf16_table(text)
            back = {u: c for c, u in enumerate(table)}
            spans = []
            for s in record.get("spans") or []:
                if s["start"] in back and s["end"] in back:
                    spans.append((back[s["start"]], back[s["end"]], s["label"]))
            docs.append({"record": record, "text": text, "spans": sorted(spans), "table": table})
    return docs


# ---- windows of tokens ----


def windows(tokenizer, docs):
    """Every document cut into overlapping windows, each token with its character offsets."""
    out = []
    for d, doc in enumerate(docs):
        enc = tokenizer(
            doc["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        for w in range(len(enc["input_ids"])):
            seq = enc.sequence_ids(w)
            offsets = [tuple(o) if seq[i] is not None else None for i, o in enumerate(enc["offset_mapping"][w])]
            out.append({"doc": d, "input_ids": enc["input_ids"][w], "offsets": offsets})
    return out


def tag_window(win, spans, tag_ids):
    """B- for the token a span starts in, I- for the rest of it, O elsewhere, -100 for special tokens."""
    labels = []
    for off in win["offsets"]:
        if off is None or off[0] == off[1]:
            labels.append(-100)
            continue
        s, e = off
        tag = "O"
        for a, b, label in spans:
            if s < b and e > a:
                tag = ("B-" if s <= a else "I-") + label
                break
        labels.append(tag_ids.get(tag, tag_ids["O"]))
    return labels


def batches(items, size, pad_id, shuffle_with=None):
    order = list(range(len(items)))
    if shuffle_with is not None:
        shuffle_with.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [items[j] for j in order[i : i + size]]
        width = max(len(c["input_ids"]) for c in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        for r, c in enumerate(chunk):
            n = len(c["input_ids"])
            ids[r, :n] = torch.tensor(c["input_ids"])
            mask[r, :n] = 1
            if "labels" in c:
                labels[r, :n] = torch.tensor(c["labels"])
        yield chunk, ids, mask, labels


# ---- predicting and scoring ----


def predict(model, tokenizer, docs, device):
    """Each document's spans, as (start, end, label) in code points."""
    model.eval()
    wins = windows(tokenizer, docs)
    best = [dict() for _ in docs]  # per doc: (start, end) -> (distance from a window edge, tag)
    with torch.no_grad():
        for chunk, ids, mask, _ in batches(wins, BATCH_SIZE * 2, tokenizer.pad_token_id):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            tags = logits.argmax(-1).cpu().tolist()
            for r, win in enumerate(chunk):
                content = [i for i, o in enumerate(win["offsets"]) if o is not None and o[0] < o[1]]
                if not content:
                    continue
                first, last = content[0], content[-1]
                for i in content:
                    key = win["offsets"][i]
                    edge = min(i - first, last - i)
                    seen = best[win["doc"]].get(key)
                    if seen is None or edge > seen[0]:
                        best[win["doc"]][key] = (edge, model.config.id2label[tags[r][i]])
    found = []
    for tokens in best:
        spans = []
        current = None
        for (s, e), (_, tag) in sorted(tokens.items()):
            if tag == "O":
                if current:
                    spans.append(tuple(current))
                current = None
                continue
            kind, label = tag.split("-", 1)
            if current and kind == "I" and current[2] == label:
                current[1] = e
            else:
                if current:
                    spans.append(tuple(current))
                current = [s, e, label]
        if current:
            spans.append(tuple(current))
        found.append(spans)
    return found


def score(docs, found, labels):
    """Exact-span precision, recall and F1 per label and over all, the same definition mlops.nlp.evaluate uses."""
    counts = {label: [0, 0, 0] for label in labels}  # true positives, predicted, gold
    for doc, spans in zip(docs, found):
        gold = set(doc["spans"])
        for a, b, label in gold:
            counts.setdefault(label, [0, 0, 0])[2] += 1
        for span in set(spans):
            counts.setdefault(span[2], [0, 0, 0])[1] += 1
            if span in gold:
                counts[span[2]][0] += 1

    def prf(tp, pred, gold):
        p = tp / pred if pred else 0.0
        r = tp / gold if gold else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    out = {}
    for label, (tp, pred, gold) in counts.items():
        if not gold and not pred:
            continue
        p, r, f = prf(tp, pred, gold)
        out[f"precision.{label}"], out[f"recall.{label}"], out[f"f1.{label}"] = p, r, f
    total = [sum(c[i] for c in counts.values()) for i in range(3)]
    out["precision.micro"], out["recall.micro"], out["f1.micro"] = prf(*total)
    return out


def write_predictions(path, docs, found):
    with open(path, "w", encoding="utf-8") as f:
        for doc, spans in zip(docs, found):
            record = {k: v for k, v in doc["record"].items() if k not in ("spans", "reviewed")}
            t = doc["table"]
            record["spans"] = [{"start": t[a], "end": t[b], "label": label, "origin": "model"} for a, b, label in spans]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---- training ----


def main():
    global EPOCHS, LEARNING_RATE, BATCH_SIZE, MAX_LENGTH
    ap = argparse.ArgumentParser(description="Train a span tagger on JSONL records.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--test")
    ap.add_argument("--out", default="model")
    ap.add_argument("--metrics", default="metrics.json")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--labels", help="Comma-separated labels; the training file's labels when absent.")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = ap.parse_args()
    EPOCHS, LEARNING_RATE, BATCH_SIZE, MAX_LENGTH = args.epochs, args.lr, args.batch_size, args.max_length

    started = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train = read_jsonl(args.train)
    if not train:
        answer({"ok": False, "reason": f"{args.train} has no documents."}, 1)
    val = read_jsonl(args.val) if args.val and os.path.exists(args.val) else []
    test = read_jsonl(args.test) if args.test else []
    labels = [x for x in args.labels.split(",") if x] if args.labels else sorted({s[2] for d in train for s in d["spans"]})
    if not labels:
        answer({"ok": False, "reason": f"{args.train} has no labelled span to learn from."}, 1)
    tags = ["O"] + [f"{k}-{label}" for label in labels for k in ("B", "I")]
    tag_ids = {t: i for i, t in enumerate(tags)}

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train.log"), "w", encoding="utf-8")

    def say(line):
        log.write(f"{time.time() - started:8.1f}s  {line}\n")
        log.flush()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not getattr(tokenizer, "is_fast", False):
        answer({"ok": False, "reason": f"{args.model} has no fast tokenizer, and span offsets need one."}, 1)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(tags),
        id2label=dict(enumerate(tags)),
        label2id=tag_ids,
    ).to(device)

    items = windows(tokenizer, train)
    for item in items:
        item["labels"] = tag_window(item, train[item["doc"]]["spans"], tag_ids)
    steps = max(1, -(-len(items) // BATCH_SIZE) * EPOCHS)
    warm = int(steps * WARMUP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
    )
    say(f"{len(train)} documents, {len(items)} windows, {len(labels)} labels, {args.model} on {device}, {steps} steps")

    shuffler = random.Random(SEED)
    best_f1, best_epoch, loss = -1.0, 0, 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n, skipped = 0.0, 0, 0
        for _, ids, mask, gold in batches(items, BATCH_SIZE, tokenizer.pad_token_id, shuffler):
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=gold.to(device))
            # A step whose loss or gradient is not a number is skipped, the way a
            # gradient scaler skips one: applied, it turns every weight into NaN
            # (measured once in five runs on mps, 2026-10-21).
            if torch.isfinite(out.loss):
                out.loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(out.loss) and torch.isfinite(norm):
                optimizer.step()
                total, n = total + out.loss.item(), n + 1
            else:
                skipped += 1
            schedule.step()
            optimizer.zero_grad()
        if skipped:
            say(f"epoch {epoch}: skipped {skipped} steps whose loss or gradient was not a number")
        if skipped * 10 > n + skipped:
            answer({"ok": False, "reason": f"The loss was not a number in {skipped} of {n + skipped} steps of epoch {epoch} on {device}; set DEVICE = \"cpu\" or a lower LEARNING_RATE in the script's configuration and train again."}, 1)
        loss = total / max(1, n)
        if val:
            f1 = score(val, predict(model, tokenizer, val, device), labels)["f1.micro"]
            say(f"epoch {epoch}: loss {loss:.4f}, validation F1 {f1:.4f}")
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                model.save_pretrained(args.out)
        else:
            say(f"epoch {epoch}: loss {loss:.4f}")
    if not val:
        best_epoch = EPOCHS
        model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "ivyx-ner.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "tags": tags, "baseModel": args.model, "maxLength": MAX_LENGTH, "stride": STRIDE, "epoch": best_epoch}, f, indent=2)

    model = AutoModelForTokenClassification.from_pretrained(args.out).to(device)
    metrics = {"train.loss": loss, "epochs": EPOCHS, "best_epoch": best_epoch}
    predictions = []
    for part, docs in (("val", val), ("test", test)):
        if not docs:
            continue
        found = predict(model, tokenizer, docs, device)
        for name, value in score(docs, found, labels).items():
            metrics[f"{part}.{name}"] = value
        path = os.path.join(args.out, f"predictions.{part}.jsonl")
        write_predictions(path, docs, found)
        predictions.append(path)
    metrics["seconds"] = round(time.time() - started, 1)
    metrics = finite(metrics)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    say(f"done: {json.dumps({k: round(v, 4) for k, v in metrics.items() if k.endswith('.micro')})}")
    answer({"ok": True, "out": args.out, "device": device, "model": args.model, "labels": labels, "metrics": metrics, "predictions": predictions, "seconds": metrics["seconds"]})


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as err:  # the step says this sentence on the card
        answer({"ok": False, "reason": f"{type(err).__name__}: {err}"}, 1)
