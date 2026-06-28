"""Sutura benchmark wrappers and the synthetic tear-deformation generator.

The optional baseline wrappers (paste2_baseline, gpsa_baseline, stalign_baseline)
each guard their optional third-party imports, so importing this package never
fails just because PASTE2 / GPSA / STalign are not installed. `deformation`
(apply_warp) has no optional dependencies and is always importable.
"""

from . import deformation

__all__ = ["deformation"]
