#!/usr/bin/env bash
# Figure 1 — registration error vs tear severity (all methods).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -c "import sys; sys.path.insert(0, '$HERE/figures'); \
import make_figures as m; m.fig1(m.FIGDIR / 'fig1_benchmark.png')"
echo "wrote $HERE/figures/fig1_benchmark.png"
