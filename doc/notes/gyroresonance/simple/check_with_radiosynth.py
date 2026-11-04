# radiosynth - From MHD to MHz: checks gyroresonance_simple.py against radiosynth
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Puts the model of gyroresonance_simple.py on a radiosynth grid, computes the
image with radiosynth.make_gr_image (seen from above, parallel rays) and compares
it with the simple code.

Usage:  python check_with_radiosynth.py            (from this folder of the repository)
        python check_with_radiosynth.py /path/to/radiosynth   (folder containing radiosynth/)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gyroresonance_simple as gs                                     # noqa: E402


def compare(freqs=(5e9, 17e9), npix=100, nz=600, verbose=True):
    """Return a list of dicts (one per frequency) with the results of both codes."""
    import radiosynth as rs
    gs.NPIX, gs.NZ = npix, nz
    ds = gs.Z_TOP / nz
    x = np.linspace(-72, 72, 145)
    z = np.linspace(0, gs.Z_TOP, 301)
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
    bx, by, bz = gs.magnetic_field(X, Y, Z)
    grid = rs.MHDGrid(x, x, z, gs.density(Z) * 1e6, gs.temperature(Z), r_body=0.0,
                      length_unit=1e6, dtype=np.float64)          # coordinates in Mm
    grid.add_field(bx, by, bz, dtype=np.float64)
    out = []
    for f in freqs:
        s = gs.gyroresonance_image(f)
        r = rs.make_gr_image(grid, f, observer=(0, 0, 1), up=(0, 1, 0), x_size=gs.FOV,
                             y_size=gs.FOV, nx=npix, ny=npix, distance_pc=4.8481e-6,
                             r_integration=1e4, s_max=6, gaunt="dulk", ds=ds, verbose=False)
        tb_r = r.brightness_temperature[0]
        res = dict(freq=f, tb_max_simple=s["tb"].max(), tb_max_rs=tb_r.max(),
                   flux_simple=s["flux_sfu"], flux_rs=r.flux_density[0] / 1e-22,
                   v_ratio=np.abs(s["tb_v"]).sum() / np.abs(r.tb_v[0]).sum(),
                   frac_bad=float((np.abs(s["tb"] - tb_r) > 0.05 * tb_r.max()).mean()))
        out.append(res)
        if verbose:
            print(f"{f / 1e9:4.0f} GHz: max T_b {res['tb_max_simple'] / 1e6:.3f} / {res['tb_max_rs'] / 1e6:.3f} MK, "
                  f"flux {res['flux_simple']:.2f} / {res['flux_rs']:.2f} sfu, |V| ratio {res['v_ratio']:.4f}, "
                  f"pixels off by > 5 %: {100 * res['frac_bad']:.1f} %   (simple / radiosynth)")
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "..")
    sys.path.insert(0, os.path.abspath(root))
    compare()
