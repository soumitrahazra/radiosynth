# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Machinery shared by all emission mechanisms.

constants.py  physical constants, critical density, plasma frequency
grid.py       MHDGrid (Cartesian) and SphericalGrid: n_e, T_e and (optionally)
              the magnetic field, with trilinear sampling
readers.py    BATS-R-US Tecplot / IDL readers, scattered-point resampling
pluto.py      PLUTO .dbl/.flt/.vtk readers (built-in or pyPLUTO)
contrib/      the pyPLUTO module distributed with PLUTO
raytrace.py   refractive ray tracer (BATS-R-US ModRadioWaveImage algorithm)
straight.py   straight-ray sampler for polarized transfer (gyroresonance)
emission.py   EmissionModel: forwards to the registered mechanism
mechanism.py  mechanism registry
image.py      image-plane geometry, make_radio_image, RadioImage
figures.py    figures, solar ephemeris/orientation, comparison with observations
stellar.py    distant observers, spectra, rotational modulation
fitsio.py     minimal FITS reader/writer
cli.py        command line (python -m radiosynth)
"""
