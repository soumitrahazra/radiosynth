# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""The teaching code doc/notes/gyroresonance/simple/gyroresonance_simple.py and
radiosynth.make_gr_image must give the same image of the same model
(low resolution, a few seconds)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "doc", "notes", "gyroresonance", "simple"))
from check_with_radiosynth import compare                              # noqa: E402

results = []
for r in compare(freqs=(5e9, 17e9), npix=40, nz=300, verbose=False):
    ok = (abs(r["flux_simple"] / r["flux_rs"] - 1) < 2e-3
          and abs(r["tb_max_simple"] / r["tb_max_rs"] - 1) < 1e-2
          and abs(r["v_ratio"] - 1) < 1e-2 and r["frac_bad"] < 0.01)
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {r['freq'] / 1e9:g} GHz: flux {r['flux_simple']:.3f} vs "
          f"{r['flux_rs']:.3f} sfu, max T_b {r['tb_max_simple'] / 1e6:.3f} vs {r['tb_max_rs'] / 1e6:.3f} MK, "
          f"|V| ratio {r['v_ratio']:.4f}")
print(f"Summary: {sum(results)} / {len(results)} passed")
sys.exit(0 if all(results) else 1)
