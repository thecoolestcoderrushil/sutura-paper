#!/usr/bin/env bash
# Regenerate all paper figures from the result CSVs in results/.
# Paths are resolved relative to this script, so it works from any cwd.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/figures/make_figures.py"
echo "Figures written to $HERE/figures/"
