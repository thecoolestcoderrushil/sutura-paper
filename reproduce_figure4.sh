#!/usr/bin/env bash
# Figure 4 — Sutura architecture schematic.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -c "import sys; sys.path.insert(0, '$HERE/figures'); \
import make_figures as m; m.fig_architecture(m.FIGDIR / 'fig_architecture.png')"
echo "wrote $HERE/figures/fig_architecture.png"
