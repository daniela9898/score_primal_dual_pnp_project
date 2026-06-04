#!/usr/bin/env bash
set -euo pipefail

# Run this from the root of your repository after unzipping the package there.
ROOT="$(pwd)"

echo "Installing SNORE-PDS package into: $ROOT"
mkdir -p "$ROOT/snore_pds" "$ROOT/examples" "$ROOT/results_snore_pds"

if [[ ! -f "$ROOT/snore_pds/solver.py" ]]; then
  echo "snore_pds package files should already be present after unzip."
  echo "If not, unzip with: unzip snore_pds_package.zip -d ."
fi

python - <<'PY'
import importlib
m = importlib.import_module('snore_pds')
print('OK: imported snore_pds from', m.__file__)
PY

echo "Done. Try: python -m snore_pds.run_snore_pds_demo --help"
