"""T-REN package.

The ``QuerySearch`` model lives in ``tren/video_query_search/models.py`` and
uses plain (non-package) imports of ``model`` / ``task_utils`` — those modules
sit directly under ``tren/``. We add ``tren/`` to ``sys.path`` here so those
imports resolve whether the user installed the package or runs from source.

Weights are not bundled with the code (~4.6 GB). Download them with
``scripts/download_tren_weights.sh``; they land under ``tren/weights/``.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(_HERE, "video_query_search")):
    if p not in sys.path:
        sys.path.insert(0, p)
