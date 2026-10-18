#!/bin/sh
# check.sh — refuses a tree that carries a machine trace or a secret.
# Run from the repository root; exits 1 and prints the lines when it finds one.
cd "$(dirname "$0")/.." || exit 2
pattern='/Users/|/private/tmp|/home/[a-z]|sezgin|(^|[^A-Za-z])sk-[A-Za-z0-9_-]{20,}|REGISTRY_TOKEN|api[_-]?key"?\s*[:=]\s*"?[A-Za-z0-9]{16,}'
if grep -rEn "$pattern" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=node_modules --exclude=check.sh . ; then
  echo "check: machine traces or secrets above" >&2
  exit 1
fi
echo "check: clean ($(find . -type f -not -path './.git/*' | wc -l | tr -d ' ') files, $(du -sh -I .git . 2>/dev/null | cut -f1))"
