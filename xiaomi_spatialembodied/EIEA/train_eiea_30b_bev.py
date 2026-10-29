#!/usr/bin/env python3
"""Deprecated alias for :mod:`training.train_eiea_30b`.

The old ``train_eiea_30b_bev.py`` imported ``train_e2e``/``prep`` from the
private experiment tree and never trained a BEV or depth cue.  The unified
30B trainer now handles box + BEV map + depth (and optional flow), random
masking and noise, so this file is kept only as a backwards-compatible entry
point.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_eiea_30b import main

if __name__ == "__main__":
    print("train_eiea_30b_bev.py is deprecated; forwarding to "
          "train_eiea_30b.py (box + BEV + depth)", file=sys.stderr)
    main()
