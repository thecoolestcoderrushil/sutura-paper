#!/usr/bin/env bash
# Figure 3 — tear vs smooth magnitude control.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -c "import sys; sys.path.insert(0, '$HERE/figures'); \
import make_figures as m; m.fig1(m.FIGDIR / 'fig3_magnitude_control.png', add_smooth=True)"
echo "wrote $HERE/figures/fig3_magnitude_control.png"
