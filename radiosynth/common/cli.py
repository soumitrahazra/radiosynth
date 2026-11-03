# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line interface.

Example
-------
python -m radiosynth 3d__mhd_box.dat --freq "80MHz, 150MHz, 17GHz" \
    --observer -215 0 0 --size 6 6 --npix 256 256 --workers 8 \
    --out radio_n001000 --plot

python -m radiosynth 3d__mhd_ar.dat --mechanism gyroresonance --freq "5GHz, 17GHz" \
    --size 0.5 0.5 --npix 200 200 --out gr_ar --plot
"""
import argparse
import sys

import numpy as np

from .readers import load_mhd
from .image import make_radio_image
from .mechanism import available_mechanisms, get_mechanism


def main(argv=None):
    ap = argparse.ArgumentParser(prog="radiosynth",
                                 description="Synthetic radio images from MHD output: "
                                             "free-free/simplistic (refractive ray tracing, "
                                             "BATS-R-US algorithm) or gyroresonance + free-free "
                                             "(straight rays, Stokes I and V)")
    ap.add_argument("input", nargs="?", help="MHD output: .dat/.tec Tecplot ASCII, .plt (pytecplot), "
                                  ".out BATS-R-US IDL, a PLUTO output directory (grid.out + "
                                  "dbl.out/flt.out) or PLUTO .vtk, or .npz saved by --save-grid")
    ap.add_argument("--freq", help="e.g. '42.3MHz, 125MHz, 17GHz'")
    ap.add_argument("--list-mechanisms", action="store_true",
                    help="list the emission mechanisms and exit")
    ap.add_argument("--observer", nargs=3, type=float, default=[-215.0, 0.0, 0.0],
                    metavar=("X", "Y", "Z"), help="observer position [R_sun]")
    ap.add_argument("--date", help="observation time, e.g. 2014-11-01T03:00: places the observer "
                                   "at Earth (B0, L0, distance) instead of --observer")
    ap.add_argument("--frame", default="carrington", choices=["carrington", "stonyhurst"],
                    help="with --date: model longitude = Carrington longitude, or 0 facing Earth")
    ap.add_argument("--phi-offset", type=float, default=0.0,
                    help="with --date: extra longitude offset of the model frame [deg]")
    st = ap.add_argument_group("stars")
    st.add_argument("--r-star", type=float, help="stellar radius in R_sun; coordinates are "
                                                    "then in stellar radii (BATS-R-US #STAR runs)")
    st.add_argument("--distance-pc", type=float, help="distance of the star [pc]: parallel rays, "
                                                         "flux at that distance (prints mJy)")
    st.add_argument("--inclination", type=float, help="inclination of the rotation axis (z) to "
                                                         "the line of sight [deg]; sets the view")
    st.add_argument("--phase", type=float, default=0.0, help="rotational phase [deg]")
    ap.add_argument("--size", nargs=2, type=float, default=[6.0, 6.0],
                    metavar=("XSIZE", "YSIZE"), help="image size [R_sun]")
    ap.add_argument("--npix", nargs=2, type=int, default=[128, 128], metavar=("NX", "NY"))
    ap.add_argument("--up", nargs=3, type=float, default=[0.0, 0.0, 1.0],
                    help="vector defining image 'up' (BATS-R-US uses +z)")
    ap.add_argument("--mechanism", "--emission", dest="emission", default="bremsstrahlung",
                    choices=["bremsstrahlung", "simplistic", "gyroresonance", "gyrosynchrotron",
                             "combined"],
                    help="emission mechanism (folder of radiosynth); gyroresonance, "
                         "gyrosynchrotron and combined (ff + gr + gs in one transfer) read "
                         "the magnetic field and use straight rays")
    ap.add_argument("--gaunt", default=None,
                    help="Gaunt factor (number) or 'dulk' (default 10; 'dulk' for gyroresonance)")
    gr = ap.add_argument_group("gyroresonance")
    gr.add_argument("--gr-mechanisms", default="gr+ff", choices=["gr+ff", "gr", "ff"],
                    help="layers + free-free (default), layers only, or magnetised free-free")
    gr.add_argument("--s-max", type=int, default=10, help="highest harmonic")
    gr.add_argument("--coupling", default="weak", choices=["weak", "strong"],
                    help="mode coupling through quasi-transverse regions")
    gr.add_argument("--ds", type=float, help="step along the rays (default half the smallest cell)")
    gr.add_argument("--t-body", type=float, help="brightness temperature of the body [K]")
    gr.add_argument("--b-unit", help="unit of B in the file if not given there: G, nT, T "
                                     "or gauss per unit")
    gs = ap.add_argument_group("gyrosynchrotron (fast GS codes; build once with "
                               "python -m radiosynth.gyrosynchrotron.library --build)")
    gs.add_argument("--nb", type=float, default=1e6,
                    help="fast-electron density [cm^-3] (peak value of the rule)")
    gs.add_argument("--nb-rule", default="sphere",
                    choices=["sphere", "uniform", "thermal-fraction", "current"],
                    help="where the fast electrons are: gaussian blob (--nb-center, "
                         "--nb-radius), everywhere, n_b = fraction x n_e (--nb as the "
                         "fraction), or proportional to |J| (n_b = nb at max |J|)")
    gs.add_argument("--nb-center", type=float, nargs=3, metavar=("X", "Y", "Z"))
    gs.add_argument("--nb-radius", type=float, default=0.02, help="blob radius [R_sun]")
    gs.add_argument("--delta", type=float, default=4.0, help="power-law index")
    gs.add_argument("--emin-kev", type=float, default=10.0)
    gs.add_argument("--emax-mev", type=float, default=10.0)
    gs.add_argument("--pitch", default="iso", choices=["iso", "gau", "elc", "gab", "sga"])
    gs.add_argument("--loss-cone", type=float, default=60.0,
                    help="loss-cone angle / beam direction [deg]")
    gs.add_argument("--dmu", type=float, default=0.2, help="loss-cone / beam width")
    gs.add_argument("--gs-coupling", default="exact", choices=["exact", "weak", "strong"])
    gs.add_argument("--gs-library", help="path of the compiled GS codes library")
    gs.add_argument("--gs-diagnostics", action="store_true",
                    help="also compute free-free alone (fast-electron fraction map)")
    gs.add_argument("--ff-refraction", action="store_true",
                    help="combined: correct the free-free part for refraction (refractive "
                         "ray tracer); gyroresonance and gyrosynchrotron stay on straight rays")
    gs.add_argument("--include", nargs="+", default=["ff", "gr", "gs"],
                    choices=["ff", "gr", "gs"],
                    help="combined: mechanisms to include (default all three; 'gs' needs --nb > 0)")
    ap.add_argument("--transfer", default="thin", choices=["thin", "full"],
                    help="thin: I=sum(kappa B ds) as in BATS-R-US; full: with absorption")
    ap.add_argument("--no-refraction", action="store_true", help="straight lines of sight")
    ap.add_argument("--compare-refraction", action="store_true",
                    help="bremsstrahlung/simplistic: compute the image with refracted rays and "
                         "with straight lines of sight; writes PREFIX_refracted.*, "
                         "PREFIX_straight.* and PREFIX_refraction.png")
    ap.add_argument("--kappa-over-mu", action="store_true",
                    help="use kappa/mu (refractive-index correction to absorption)")
    ap.add_argument("--r-integration", type=float, default=None)
    ap.add_argument("--r-body", type=float, default=None,
                    help="inner boundary [R_sun] (default 1, or PLUTO inner radius)")
    ap.add_argument("--step-factor", type=float, default=1.0)
    ap.add_argument("--tolerance", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=1)
    # variable selection
    ap.add_argument("--density-var")
    ap.add_argument("--density-unit", help="e.g. g/cm^3, kg/m^3, cm^-3, m^-3")
    ap.add_argument("--te-var")
    ap.add_argument("--pe-var")
    ap.add_argument("--p-var")
    ap.add_argument("--te-const", type=float)
    ap.add_argument("--mu", type=float, default=1.0,
                    help="mean mass per electron in m_p (n_e = rho/(mu m_p)); BATS-R-US uses 1")
    ap.add_argument("--resample", type=int, nargs="+",
                    help="resample scattered/AMR points to this uniform shape (1 or 3 ints)")
    ap.add_argument("--resample-box", type=float, nargs=6,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    ap.add_argument("--resample-method", default="linear", choices=["linear", "nearest"])
    ap.add_argument("--save-grid", help="save the n_e/T_e grid as .npz for fast reuse")
    sy = ap.add_argument_group("1D (spherically symmetric) models")
    sy.add_argument("--profile", action="store_true",
                    help="INPUT is a text table of a 1D model (one radius per line), "
                         "assumed spherically symmetric")
    sy.add_argument("--profile-cols", type=int, nargs="+", default=[0, 1, 2], metavar="COL",
                    help="columns of r, density, T_e (default 0 1 2); "
                         "give 2 columns for no temperature")
    sy.add_argument("--profile-density", default="ne", choices=["ne", "rho"])
    sy.add_argument("--profile-density-unit", default=None,
                    help="cm^-3 / m^-3 for ne (default cm^-3), g/cm^3 / kg/m^3 for rho "
                         "(default g/cm^3)")
    sy.add_argument("--profile-r-unit", type=float, default=1.0,
                    help="coordinate units (R_sun or R_star) per file unit of r")
    sy.add_argument("--symmetric", action="store_true",
                    help="spherically symmetric model: trace one row of rays and rotate "
                         "(fast); writes the T_b profile (PREFIX_profile.txt/.png)")
    sy.add_argument("--n-p", type=int, default=1000, help="number of rays for --symmetric")
    pl = ap.add_argument_group("PLUTO input")
    pl.add_argument("--nout", type=int, default=-1, help="PLUTO output number (default last)")
    pl.add_argument("--pluto-format", choices=["dbl", "flt"], help="default: autodetect")
    pl.add_argument("--definitions", help="path to definitions.h (for UNIT_*)")
    pl.add_argument("--unit-density", type=float, help="g/cm^3")
    pl.add_argument("--unit-length", type=float, help="cm (R_sun = 6.96e10)")
    pl.add_argument("--unit-velocity", type=float, help="cm/s")
    pl.add_argument("--mu-e", type=float, default=1.0, help="n_e = rho/(mu_e m_p)")
    pl.add_argument("--pypluto", nargs="?", const="", default=None, metavar="DIR",
                    help="read PLUTO data with pyPLUTO (optionally give the directory "
                         "containing pyPLUTO.py)")
    pl.add_argument("--mmw", type=float, default=0.5,
                    help="mean molecular weight for T = p mu m_p/(rho k)")
    # output
    ap.add_argument("--out", default="radio_image", help="output prefix")
    ap.add_argument("--formats", nargs="+", default=["npz", "tec"],
                    choices=["npz", "tec", "fits"])
    ap.add_argument("--plot", action="store_true", help="save PNG of T_b and intensity")
    a = ap.parse_args(argv)
    if a.list_mechanisms:
        for name in available_mechanisms():
            print(f"{name:16s} {get_mechanism(name).description}")
        return 0
    if a.input is None or a.freq is None:
        ap.error("the input file and --freq are required")
    use_gr = a.emission == "gyroresonance"
    use_gs = a.emission == "gyrosynchrotron"
    use_cb = a.emission == "combined"
    if a.compare_refraction and (use_gr or use_gs or use_cb):
        ap.error("--compare-refraction is available for bremsstrahlung and simplistic only "
                 "(gyroresonance, gyrosynchrotron and combined use straight rays)")

    rs = None
    if a.resample:
        rs = tuple(a.resample) if len(a.resample) == 3 else (a.resample[0],) * 3
    box = None
    if a.resample_box:
        b = a.resample_box
        box = ((b[0], b[1]), (b[2], b[3]), (b[4], b[5]))
    if a.profile:
        from .symmetric import read_profile
        c = a.profile_cols
        du = a.profile_density_unit or ("cm^-3" if a.profile_density == "ne" else "g/cm^3")
        if use_gr or use_gs or use_cb:
            ap.error("--profile (1D models): use --mechanism bremsstrahlung or simplistic")
        grid = read_profile(a.input, r_col=c[0], density_col=c[1],
                            te_col=c[2] if len(c) > 2 else None, r_unit=a.profile_r_unit,
                            density=a.profile_density, density_unit=du, mu_e=a.mu_e,
                            r_body=a.r_body, length_unit=(a.r_star or 1.0) * 6.96e8)
        if len(c) <= 2 and a.te_const is not None:
            grid._flat[:, 4] = a.te_const
    else:
      grid = load_mhd(a.input, density_var=a.density_var, density_unit=a.density_unit,
                      te_var=a.te_var, pe_var=a.pe_var, p_var=a.p_var, te_const=a.te_const,
                      mu=a.mu, r_body=a.r_body, resample_shape=rs, resample_box=box,
                      resample_method=a.resample_method, nout=a.nout, fmt=a.pluto_format,
                      definitions=a.definitions, unit_density=a.unit_density,
                      unit_length=a.unit_length, unit_velocity=a.unit_velocity,
                      mu_e=a.mu_e, mean_mol_weight=a.mmw,
                      pluto_reader="pypluto" if a.pypluto is not None else None,
                      pypluto_path=a.pypluto or None, r_star=a.r_star,
                      with_b=(use_gr or use_gs or use_cb) or None,
                      b_unit=(float(a.b_unit) if a.b_unit and a.b_unit[0].isdigit()
                              else a.b_unit))
    if a.save_grid:
        grid.to_npz(a.save_grid)
    g_ = a.gaunt or ("dulk" if (use_gr or use_cb) else "10")
    gaunt = g_ if g_.lower() == "dulk" else float(g_)
    observer = a.observer
    if a.date:
        from .figures import observer_for_date
        observer = observer_for_date(a.date, phi_offset_deg=a.phi_offset, frame=a.frame)
    if a.inclination is not None:
        from .stellar import stellar_observer
        observer = stellar_observer(a.inclination, a.phase)
    if a.symmetric:
        from .symmetric import make_symmetric_image, plot_profile
        if use_gr or use_gs or use_cb:
            ap.error("--symmetric (1D models): use --mechanism bremsstrahlung or simplistic")
        mech_kw = dict(gaunt=gaunt, transfer=a.transfer, no_refraction=a.no_refraction,
                       step_factor=a.step_factor, tolerance=a.tolerance,
                       kappa_over_mu=a.kappa_over_mu)
        if a.r_integration is not None:
            mech_kw["r_integration"] = a.r_integration
        img = make_symmetric_image(grid, a.freq, mechanism=a.emission, observer=observer, x_size=a.size[0], y_size=a.size[1],
                                   nx=a.npix[0], ny=a.npix[1], n_p=a.n_p,
                                   distance_pc=a.distance_pc, **mech_kw)
        pr = img.profile
        np.savetxt(a.out + "_profile.txt",
                   np.column_stack([pr["p"]] + [pr["tb"][k] for k in range(len(img.freqs))]),
                   header="p [R] " + " ".join(f"Tb({f:.6g}Hz)[K]" for f in img.freqs))
        plot_profile(img, path=a.out + "_profile.png")
        print("flux density from the profile [W m^-2 Hz^-1]:", " ".join(f"{v:.4e}" for v in pr["flux"]))
    elif a.compare_refraction:
        from .image import compare_refraction
        res = compare_refraction(grid, a.freq, observer=observer, x_size=a.size[0],
                                 y_size=a.size[1], nx=a.npix[0], ny=a.npix[1], up=a.up,
                                 emission=a.emission, gaunt=gaunt, transfer=a.transfer,
                                 r_integration=a.r_integration, n_workers=a.workers,
                                 step_factor=a.step_factor, tolerance=a.tolerance,
                                 kappa_over_mu=a.kappa_over_mu, distance_pc=a.distance_pc,
                                 path=a.out + "_refraction.png")
        for tag in ("refracted", "straight"):
            im = res[tag]
            if "npz" in a.formats:
                im.save_npz(f"{a.out}_{tag}.npz")
            if "tec" in a.formats:
                im.save_tecplot(f"{a.out}_{tag}.dat")
            if "fits" in a.formats:
                im.save_fits(f"{a.out}_{tag}_Tb.fits", quantity="tb")
        print("written:", a.out + "_refracted.*, " + a.out + "_straight.*, " +
              a.out + "_refraction.png")
        return 0
    el = None
    if use_gs or (use_cb and "gs" in a.include and a.nb > 0):
        from ..gyrosynchrotron import (FastElectrons, uniform, fraction_of_thermal, power_of,
                                       in_sphere)
        if a.nb_rule == "sphere":
            if a.nb_center is None:
                ap.error("--nb-rule sphere needs --nb-center X Y Z")
            rule = in_sphere(a.nb_center, a.nb_radius, a.nb)
        elif a.nb_rule == "uniform":
            rule = uniform(a.nb)
        elif a.nb_rule == "thermal-fraction":
            rule = fraction_of_thermal(a.nb)
        else:
            from .fields import current_density
            j = current_density(grid)
            rule = power_of("j", a.nb, float(j.max()), 1.0, floor=0.1)
        el = FastElectrons(nb=rule, delta=a.delta, emin_mev=a.emin_kev / 1e3,
                           emax_mev=a.emax_mev, pitch=a.pitch, alpha_c=a.loss_cone, dmu=a.dmu)
    if a.symmetric:
        pass                                   # image already made from the 1D profile
    elif use_cb:
        from ..combined import make_combined_image
        img = make_combined_image(grid, a.freq, el, observer=observer, x_size=a.size[0],
                                  y_size=a.size[1], nx=a.npix[0], ny=a.npix[1], up=a.up,
                                  include=a.include, gaunt=gaunt, s_max=a.s_max,
                                  coupling=a.coupling, ds=a.ds, t_body=a.t_body,
                                  r_integration=a.r_integration, distance_pc=a.distance_pc,
                                  library=a.gs_library, ff_refraction=a.ff_refraction,
                                  trace_kw=dict(n_workers=a.workers, step_factor=a.step_factor,
                                                tolerance=a.tolerance))
    elif use_gs:
        from ..gyrosynchrotron import make_gs_image
        img = make_gs_image(grid, a.freq, el, observer=observer, x_size=a.size[0],
                            y_size=a.size[1], nx=a.npix[0], ny=a.npix[1], up=a.up,
                            coupling=a.gs_coupling, ds=a.ds, t_body=a.t_body,
                            r_integration=a.r_integration, distance_pc=a.distance_pc,
                            library=a.gs_library, diagnostics=a.gs_diagnostics,
                            abundance="dulk" if (a.gaunt or "").lower() == "dulk" else "coronal")
    elif use_gr:
        from ..gyroresonance import make_gr_image
        img = make_gr_image(grid, a.freq, observer=observer, x_size=a.size[0],
                            y_size=a.size[1], nx=a.npix[0], ny=a.npix[1], up=a.up,
                            mechanisms=a.gr_mechanisms, gaunt=gaunt, s_max=a.s_max,
                            coupling=a.coupling, ds=a.ds, t_body=a.t_body,
                            r_integration=a.r_integration, distance_pc=a.distance_pc)
    else:
        img = make_radio_image(grid, a.freq, observer=observer, x_size=a.size[0],
                               y_size=a.size[1], nx=a.npix[0], ny=a.npix[1], up=a.up,
                               emission=a.emission, gaunt=gaunt, transfer=a.transfer,
                               no_refraction=a.no_refraction, r_integration=a.r_integration,
                               n_workers=a.workers, step_factor=a.step_factor,
                               tolerance=a.tolerance, kappa_over_mu=a.kappa_over_mu,
                               distance_pc=a.distance_pc)
    if "npz" in a.formats:
        img.save_npz(a.out + ".npz")
    if "tec" in a.formats:
        img.save_tecplot(a.out + ".dat")
    if "fits" in a.formats:
        img.save_fits(a.out + "_I.fits")
        img.save_fits(a.out + "_Tb.fits", quantity="tb")
        if use_gr or use_gs or use_cb:
            img.save_fits(a.out + "_V.fits", quantity="v")
            img.save_fits(a.out + "_TbV.fits", quantity="tb_v")
    if a.plot:
        img.plot("tb", path=a.out + "_Tb.png")
        img.plot("intensity", path=a.out + "_I.png")
        if use_gr or use_gs or use_cb:
            img.plot("pol", path=a.out + "_pol.png")
        if use_gr or use_cb:
            img.plot("harmonic", path=a.out + "_harmonic.png")
        if use_cb:
            img.plot("mechanism", path=a.out + "_mechanism.png")
        if use_gs and a.gs_diagnostics:
            img.plot("gs_fraction", path=a.out + "_gsfraction.png")
    if a.distance_pc:
        for f, fl in zip(img.freqs, img.flux_density_mjy):
            print(f"flux density at {a.distance_pc:g} pc, {f / 1e9:.4g} GHz: {fl:.4g} mJy")
    print("written:", a.out + ".*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
