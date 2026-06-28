#!/usr/bin/env bash
# Figure 2 — leave-one-donor-out generalization (3 folds).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -c "import sys; sys.path.insert(0, '$HERE/figures'); \
import make_figures as m; m.fig2(m.FIGDIR / 'fig2_lodo.png')"
echo "wrote $HERE/figures/fig2_lodo.png"
