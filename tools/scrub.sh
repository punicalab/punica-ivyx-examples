#!/bin/sh
# scrub.sh <source-project> <target-folder> [keep-events]
#
# Copies a project that was opened in IVYX Studio into this repository with
# everything that belongs to one machine left behind: the audit trail
# (.punica/events, kept only when the third argument is "keep-events"), the
# search index, the Python environment record (absolute paths), the DuckDB
# cache, DVC's temp and cache, virtual environments, MLflow stores and
# bytecode. The settings file keeps its shape and loses the "system" block,
# which names the models of the machine it was written on.
#
# After copying, `tools/check.sh` greps the repository for machine traces.
set -e
src="$1"; dst="$2"; keep="$3"
[ -d "$src" ] || { echo "no such source: $src" >&2; exit 2; }
rm -rf "$dst"
mkdir -p "$dst"
rsync -a \
  --exclude '.punica/search' \
  --exclude '.punica/python.yaml' \
  --exclude '.punica/duckdb.json' \
  --exclude '.punica/datasets.json' \
  --exclude '.punica/learn' \
  --exclude '.punica/mlartifacts' \
  --exclude '.punica/mlflow.db' \
  --exclude '.punica/skills' \
  --exclude '.dvc/tmp' \
  --exclude '.dvc/cache' \
  --exclude '.venv' \
  --exclude 'mlruns' \
  --exclude 'mlartifacts' \
  --exclude '*.db' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude '.git' \
  --exclude 'model.pkl' \
  --exclude 'metrics.json' \
  $( [ "$keep" = "keep-events" ] || echo "--exclude .punica/events" ) \
  "$src/" "$dst/"
# The project's own git history stays behind, so DVC must not look for one:
# without this, `dvc repro` in a copy says "not a git repository".
if [ -f "$dst/.dvc/config" ] && ! grep -q "no_scm" "$dst/.dvc/config"; then
  printf '[core]\n    no_scm = true\n' >> "$dst/.dvc/config"
fi
if [ -f "$dst/.punica/settings.json" ]; then
  node -e '
    const fs = require("fs"); const p = process.argv[1];
    const j = JSON.parse(fs.readFileSync(p, "utf8"));
    delete j.system;
    fs.writeFileSync(p, JSON.stringify(j, null, 2) + "\n");
  ' "$dst/.punica/settings.json"
fi
if [ "$keep" = "keep-events" ] && [ -d "$dst/.punica/events" ]; then
  # The trail records absolute paths of the machine it was written on: the
  # workspace itself, its siblings (a trail cut from another fixture names
  # that one) and the home folder.
  ws="$(cd "$src" && pwd -P)"
  root="$(dirname "$(dirname "$ws")")"
  home="$HOME"
  for f in "$dst"/.punica/events/*.json; do
    [ -f "$f" ] || continue
    node -e '
      const fs = require("fs"); const [p, ws, root, home] = process.argv.slice(1);
      let t = fs.readFileSync(p, "utf8");
      t = t.split(ws).join("<workspace>").split(root).join("<machine>").split(home).join("<home>");
      fs.writeFileSync(p, t);
    ' "$f" "$ws" "$root" "$home"
  done
fi
echo "scrubbed $(basename "$dst"): $(find "$dst" -type f | wc -l | tr -d ' ') files, $(du -sh "$dst" | cut -f1)"
