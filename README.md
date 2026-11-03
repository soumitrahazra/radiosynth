# radiosynth — From MHD to MHz

**Synthetic radio images of MHD models of the Sun and stars, and their comparison with observations**

*Author: Soumitra Hazra · Licence: GPL-3.0-or-later · Version 2.2.0 (second version of the package)*

radiosynth turns the output of a magnetohydrodynamic (MHD) simulation of a solar or
stellar corona into the radio images, spectra and polarisation maps that a radio telescope
would see. For every pixel it follows the radio ray from the observer through the model
plasma — bending with the refractive index of the plasma where that matters — and solves
the radiative transfer of the emission along the way.

It works on **saved** model output from **BATS-R-US** (Tecplot or IDL files), **PLUTO**
(`.dbl`, `.flt`, `.vtk`; Cartesian or spherical; 3D, 2.5D or 1D) or **any other model** whose
density, temperature (and, for the magnetic mechanisms, magnetic field) you can put into
NumPy arrays — for the Sun seen from Earth, or for a star at a distance in parsec.

## What radiosynth computes

Each emission mechanism has its own folder in the package; everything they share
(readers, grids, geometry, ray tracing, figures, comparison with observations) is in
`radiosynth/common/`.

### Ray tracing with refraction

Below a few hundred MHz the corona refracts radio waves strongly: the refractive index
is μ = √ε with ε = 1 − n_e/n_cr, and rays are bent away from dense plasma and reflected
before they reach the critical density n_cr (where the plasma frequency equals the
observing frequency). radiosynth traces these curved rays with the algorithm of the
BATS-R-US radio-wave imaging module (Benkevitch et al. 2010; Moschou et al. 2018):
a Boris (CYLRAD) rotation of the ray direction, a parabolic switch near the critical
surface, step control by curvature and by distance to the critical surface, and a
step-back / Snell-reflection fallback for rays that overshoot. All rays are advanced
together in vectorised NumPy. Refraction can be switched off (`no_refraction=True`) and
the two images compared directly (`rs.compare_refraction`). The refractive tracer is used
for the simplistic and bremsstrahlung mechanisms.

### 1. Simplistic emission — `emission="simplistic"`

The ad-hoc emissivity of Benkevitch et al. (2010): source function S = 1 and
κL = (n/n_cr)² (0.5 − n/n_cr)². It needs neither temperature nor magnetic field, peaks
where n_e = n_cr/4 and vanishes at n_cr/2, so the image shows the layer just above the
critical surface. Useful to visualise refraction and to test the ray tracer.

### 2. Thermal free–free (bremsstrahlung) — `emission="bremsstrahlung"` (default)

The thermal emission of electrons scattered by ions, as in Moschou et al. (2018):
Rayleigh–Jeans source function S = 2ν²k_BT_e/c² and absorption coefficient
κ ∝ n_e² T_e^(−3/2) ν^(−2) g_ff, with a constant Gaunt factor g_ff = 10 or the
Coulomb-logarithm form of Dulk (1985) (`gaunt="dulk"`).

* `transfer="thin"`: I = Σ κ S Δl, as in BATS-R-US;
* `transfer="full"`: the exact solution of dI/ds = κ(S − I), so T_b ≤ T_e and the
  optical depth τ of every pixel is kept (recommended below ~1 GHz);
* optional κ/μ enhancement of absorption near the critical layer (`kappa_over_mu=True`).

This is the dominant emission of the quiet corona at metre and decimetre wavelengths.
It is available for 3D, 2.5D and 1D (spherically symmetric) models, with or without
refraction. In a magnetic field, the **magnetised free–free** emission of the two
magneto-ionic modes, κ_σ = κ_ff/(1 + σY|cos θ|)² with Y = ν_B/ν, gives circular
polarisation (used by the gyroresonance and combined images).

### 3. Thermal gyroresonance — `rs.make_gr_image`

Thermal electrons gyrating in the magnetic field of active regions absorb and emit
strongly in thin layers where the observing frequency is a low harmonic of the
gyrofrequency, ν = s ν_B (ν_B = 2.80 MHz × B[G]). radiosynth finds these layers along
each line of sight and uses the optical depth of each layer for the ordinary and
extraordinary modes (Zheleznyakov 1970; White & Kundu 1997; Fleishman & Kuznetsov 2014),
τ_s ∝ n_e L_B (s²/s!) (s²β_T² sin²θ/2)^(s−1) F_σ(θ), with Kirchhoff's law per mode, harmonics
up to `s_max` (default 10), together with magnetised free–free emission. The result is
Stokes I and V (circular polarisation, V > 0 right-handed in the IAU/IEEE convention),
with weak or strong mode coupling in quasi-transverse regions, the dominant harmonic and
the radius of each layer. This is the main source of active-region emission at
1–20 GHz. Straight lines of sight (refraction is negligible at these frequencies).

### 4. Gyrosynchrotron emission of fast electrons — `rs.make_gs_image`

Mildly relativistic electrons accelerated in flares emit gyrosynchrotron radiation at
many harmonics. An MHD model has no fast electrons, so they are **prescribed** with
`FastElectrons`:

* energy distribution: power law, double power law, kappa, thermal + power-law tail, ...;
* pitch-angle distribution: isotropic, loss cone, beam, ...;
* density n_b as a number or a **rule** of the local plasma — a blob, a box, a fraction of
  the thermal density, a power of |B| or of the current density |J| — or a tabulated
  distribution f(E, μ) from test-particle or Fokker–Planck results.

The emission and the polarised transfer are computed by the public **fast
gyrosynchrotron codes** (Fleishman & Kuznetsov 2010; Kuznetsov & Fleishman 2021; the
engine of GX Simulator; GPL-3.0, not bundled), which radiosynth downloads, builds and
calls as a compiled library. radiosynth supplies the lines of sight and the plasma,
field and fast-electron parameters in every voxel. Output: Stokes I and V, spectra and
the gyrosynchrotron share of the emission.

### 5. Combined image — `rs.make_combined_image`

Free–free, gyroresonance and gyrosynchrotron emission in **one** polarised radiative
transfer along each line of sight, so that each mechanism absorbs the others (for example a
thick gyroresonance layer in front of a gyrosynchrotron source). The gyrosynchrotron
optical depth and emission of each voxel are taken from the fast GS codes; optical
depths add and the source function is the optical-depth-weighted mean. You get Stokes I
and V, the share of each mechanism in every pixel and a map of the dominant one; any
subset can be selected (`include=("gr", "gs")`), and the free–free part can optionally be
corrected for refraction (`ff_refraction=True`). Gyroresonance and gyrosynchrotron are
never refracted.

### Also included

* **1D (spherically symmetric) models** — radial profiles, spectra and images from a
  single row of rays (bremsstrahlung and simplistic);
* **other stars** — coordinates in stellar radii, an observer at a distance in parsec,
  flux spectra, spectral indices, rotational modulation and VLBI-scale images;
* **comparison with observations** — solar ephemeris and orientation, helioprojective
  coordinates, overlays, statistics and figures in the style of Moschou et al. (2018);
* **command line** — `python -m radiosynth model_file --freq "80MHz, 150MHz" --plot`.

| Mechanism | Needs | Rays | Output |
|---|---|---|---|
| simplistic | n_e | refracted or straight | I |
| bremsstrahlung | n_e, T_e | refracted or straight | I (thin or full transfer), τ |
| gyroresonance (+ magnetised free–free) | n_e, T_e, **B** | straight | I, V |
| gyrosynchrotron (+ free–free) | n_e, T_e, **B**, fast electrons | straight | I, V |
| combined | n_e, T_e, **B**, (fast electrons) | straight (free–free optionally refraction-corrected) | I, V, share of each mechanism |

## Installation

```bash
git clone https://github.com/<your-account>/radiosynth.git
cd radiosynth
pip install -e .            # or: pip install -e ".[all]" for astropy and sunpy
python -m radiosynth --help
```

Without installing, you can also add the folder that contains `radiosynth/` to your
`PYTHONPATH`. For gyrosynchrotron and combined images, build the fast GS codes once
(git, make and a C++ compiler with OpenMP are needed):

```bash
python -m radiosynth.gyrosynchrotron.library --build
```

Requirements: `numpy`, `scipy`, `matplotlib`. Optional extras:

* `astropy` for FITS files; a built-in reader/writer is used without it;
* `pytecplot` for binary `.plt` files;
* `sunpy` to look up B0/L0 for a date;
* `ffmpeg` for MP4 animations.

## Quick start

```python
import radiosynth as rs

# BATS-R-US: Tecplot ASCII box/3d output, IDL .out, or binary .plt (via pytecplot)
grid = rs.load_mhd("box_mhd_4_n00155869.dat")

# PLUTO: an output directory (grid.out + dbl.out/flt.out), or a .vtk file.
# Units are read from definitions.h in that directory or its parent.
grid = rs.load_mhd("pluto_run/out", nout=-1)             # spherical or Cartesian

# Any other model: give the axes in R_sun, n_e in m^-3 and T_e in K
grid = rs.MHDGrid(x, y, z, ne, te)                        # Cartesian
grid = rs.SphericalGrid(r, theta, phi, ne, te)            # r [R_sun], colatitude, longitude

img = rs.make_radio_image(grid, "42.3MHz, 80MHz, 150MHz, 17GHz",
                          observer=(-215, 0, 0),           # R_sun, like ObsPos in PARAM.in
                          x_size=6, y_size=6, nx=256, ny=256,
                          emission="bremsstrahlung",       # or "simplistic"
                          transfer="thin",                 # BATS-R-US; or "full"
                          no_refraction=False,             # UseNoRefraction
                          n_workers=8)
img.plot("tb", path="radio_Tb.png")
img.save_npz("radio.npz"); img.save_tecplot("radio.dat"); img.save_fits("radio.fits")
print(img.flux_density_sfu, img.brightness_temperature.max())
```

Command line:

```bash
python -m radiosynth box_mhd.dat   --freq "80MHz, 150MHz, 17GHz" --size 6 6 --npix 256 256 --workers 8 --plot
python -m radiosynth pluto_run/out --freq "80MHz, 150MHz" --definitions pluto_run/definitions.h --transfer full --plot
python -m radiosynth --help     # variable names and units, resampling, PLUTO units, and more
```

## Package layout

```
radiosynth/
  common/          used by every mechanism: constants, magnetoionic, grid (+B), readers,
                   pluto, contrib/pyPLUTO, mechanism (registry), emission (interface),
                   raytrace (refractive), straight (straight rays), image, figures,
                   stellar, fitsio, cli
  simplistic/      Benkevitch et al. (2010) emissivity
  bremsstrahlung/  thermal free-free (g=10 or Dulk), two-mode magnetised free-free,
                   analytic wind solution
  gyroresonance/   harmonic-layer opacity, polarised straight-ray transfer, GRImage
  gyrosynchrotron/ fast GS codes as a library (build/load), FastElectrons
                   prescription, GSImage
  combined/        free-free + gyroresonance + gyrosynchrotron in one transfer
                   (per-voxel GS coefficients from the codes), CombinedImage
```

Each mechanism folder has `make_image(grid, frequencies, ...)`; the command line
takes `--mechanism simplistic|bremsstrahlung|gyroresonance|gyrosynchrotron|combined`. The
modules in `common/` can also be imported by their short paths (`radiosynth.constants`,
`rs.figures`, ...).

## Gyroresonance

```python
grid = rs.load_mhd("3d__mhd_ar.dat", with_b=True)     # Bx/By/Bz; b_unit="nT" if not in the file
grid = rs.load_pluto("out/", with_b=True)             # bx1..bx3 -> gauss
# or: grid = rs.MHDGrid(x, y, z, ne, te); grid.add_field(bx, by, bz)   # gauss

img = rs.make_gr_image(grid, "3GHz, 5GHz, 17GHz", observer=(-215, 0, 0),
                       x_size=0.4, y_size=0.4, nx=200, ny=200)   # mechanisms="gr+ff"
img.plot("tb"); img.plot("pol"); img.plot("harmonic")
img.stokes_v, img.polarization, img.tb_pol, img.dominant_s, img.radius, img.gr_fraction
```

```
python -m radiosynth ar.dat --mechanism gyroresonance --freq "5GHz, 17GHz" \
    --size 0.4 0.4 --npix 200 200 --out gr --plot --formats npz fits
```

Layer optical depth (x and o modes), Kirchhoff's law per mode, magnetised
free-free `kappa/(1 -/+ Y|cos|)^2`, straight rays, weak or strong mode coupling;
V > 0 is right-hand circular (IAU/IEEE). See `examples/gyroresonance_example.py`
and Chapter "Gyroresonance emission and circular polarisation" of the manual.

## Gyrosynchrotron (fast electrons)

The emission is computed by the public fast gyrosynchrotron codes
(https://github.com/kuznetsov-radio/gyrosynchrotron, GPL-3.0, **not bundled**).
Build them once (git, make, g++ with OpenMP):

```bash
python -m radiosynth.gyrosynchrotron.library --build     # -> ~/.radiosynth/gyrosynchrotron
# or set RADIOSYNTH_GSCODES=/path/to/MWTransferArr.so
```

The fast electrons are prescribed (an MHD model has none):

```python
from radiosynth.gyrosynchrotron import FastElectrons, in_sphere, fraction_of_thermal, power_of
el = FastElectrons(nb=in_sphere((-1.04, 0, 0), 0.015, 1e7),   # cm^-3, gaussian blob
                   delta=3.5, emin_mev=0.02, emax_mev=5.0, pitch="iso")
img = rs.make_gs_image(grid, "3GHz, 10GHz, 17GHz", el, x_size=0.4, y_size=0.4,
                       nx=100, ny=100, diagnostics=True)
img.plot("tb"); img.plot("pol"); img.plot("gs_fraction")
# other rules: fraction_of_thermal(1e-3), power_of("bmag", ...),
# rs.current_density(grid) then power_of("j", ...), any function of the sampled fields,
# or FastElectrons.from_array(E, mu, f, nb=...) for tabulated distributions
```

```bash
python -m radiosynth flare.dat --mechanism gyrosynchrotron --freq "5GHz, 17GHz" \
    --nb 1e7 --nb-center -1.04 0 0 --nb-radius 0.015 --delta 3.5 --size 0.4 0.4 --plot
```

radiosynth supplies the straight lines of sight and the plasma and fast-electron
parameters in every voxel; the codes do the emission and transfer (gyrosynchrotron +
free-free, exact/weak/strong mode coupling). Thermal gyroresonance is not included in
this calculation; use the gyroresonance folder for the thermal active region.

## Combined image (free-free + gyroresonance + gyrosynchrotron)

```python
el = rs.FastElectrons(nb=rs.gyrosynchrotron.in_sphere((-1.04, 0, 0), 0.015, 3e5),
                      delta=3.5, emin_mev=0.02, emax_mev=5.0)
img = rs.make_combined_image(grid, "2GHz, 5GHz, 10GHz, 17GHz", el, x_size=0.36, y_size=0.36)
img.plot("tb"); img.plot("mechanism"); img.plot("pol")
img.ff_fraction, img.gr_fraction, img.gs_fraction      # share of each process per pixel
rs.make_combined_image(grid, freqs, el, include=("gr", "gs"))   # any subset
rs.make_combined_image(grid, freqs, el, ff_refraction=True)     # free-free corrected for refraction
```

```bash
python -m radiosynth ar.dat --mechanism combined --freq "5GHz, 17GHz" --nb 3e5 \
    --nb-center -1.04 0 0 --nb-radius 0.015 --delta 3.5 --size 0.4 0.4 --plot [--ff-refraction]
```

One polarised transfer along each straight ray: optical depths add, the source function
is the optical-depth-weighted mean. The gyrosynchrotron optical depth and emission of
each voxel with fast electrons come from two one-voxel calls of the fast GS codes
(without and with incoming intensity). Gyroresonance and gyrosynchrotron use straight
rays; the free-free part can optionally be corrected for refraction
(I_p -> I_p + I_ff,p (R - 1), R = refracted/straight free-free intensity from the
refractive tracer). Without fast electrons the result equals `make_gr_image` exactly.

## Refracted rays vs straight lines of sight

For the mechanisms that use the refractive ray tracer (bremsstrahlung, simplistic):

```python
img_ref = rs.make_radio_image(grid, "80MHz")                      # refracted (default)
img_los = rs.make_radio_image(grid, "80MHz", no_refraction=True)  # straight lines of sight
res = rs.compare_refraction(grid, "80MHz, 17GHz", transfer="full", path="refraction.png")
res["flux_ratio"], res["changed"]      # refracted/straight flux; fraction of pixels changed >10%
```

```bash
python -m radiosynth corona.dat --freq "80MHz, 17GHz" --compare-refraction --out cmp
```

Gyroresonance and gyrosynchrotron are computed along straight rays only.

## 1D (spherically symmetric) models

```python
grid = rs.grid_from_profile(r, ne, te, r_body=1.0)        # r [R_sun], n_e [m^-3], T_e [K]
grid = rs.read_profile("model.txt", r_col=0, density_col=1, te_col=2,
                       density="rho", density_unit="g/cm^3", r_unit=1/6.96e10)
grid = rs.load_pluto("out1d/")                            # 1D spherical PLUTO output
img = rs.make_symmetric_image(grid, "80MHz, 300MHz, 1GHz", transfer="full", x_size=6)
img.profile["p"], img.profile["tb"], img.profile["flux"]  # T_b(p) and total flux
rs.plot_profile(img, path="profiles.png")
```

```bash
python -m radiosynth model.txt --profile --symmetric --freq "80MHz, 1GHz" --size 6 6 --out quiet
```

One row of rays (default 1000) is traced and rotated into an image: exact for a
symmetric model, with refraction on or off (`no_refraction=True`), and much faster
than the full image. Useful for spectra, radial brightness profiles, apparent sizes,
stellar winds and parameter studies; available for bremsstrahlung and simplistic
(gyroresonance and gyrosynchrotron are not offered for 1D models).

## Figures like Moschou et al. (2018)

`radiosynth.figures` reproduces the figure types of the paper:

| Paper figure | Function | What it shows |
|---|---|---|
| Fig. 2 | `figure_compare_observation` | synthetic map and observed FITS map with synthetic contours |
| Fig. 3 | `figure_single_map` | one frequency with log contours |
| Fig. 5 | `figure_time_series` | times × frequencies grid |

`animate_time_series` writes a GIF or MP4. `earth_observer(B0, L0)` and
`units="arcsec"` give solar North up, West right, in helioprojective arcsec.
`examples/moschou_figures.py` runs everything on a test corona.

```python
from radiosynth import figures as rf
img = rs.make_radio_image(grid, "17GHz", rf.earth_observer(B0, L0), 2.6, 2.6, 300, 300)
rf.figure_compare_observation(img, "norh_17GHz.fits", beam_fwhm_arcsec=10, path="fig2.png")
```

## Comparing models without solar coordinates (PLUTO) with observations

* `observer_for_date(t, phi_offset_deg=...)` places the observer at the Earth for the
  observation time. It computes B0, the Carrington L0, the P-angle and the distance,
  using sunpy if installed and Meeus' algorithm otherwise. On the command line, use
  `--date T --phi-offset X`.
* `fit_longitude_offset(...)` finds an unknown longitude offset of the model frame by
  correlating synthetic maps with an observed map. In the tests it recovered a known
  +17° offset as +16.8°.
* `compare_with_observation(...)` puts the synthetic map, smoothed to the beam, onto
  the observed pixels. It returns the disk and limb T_b, the log correlation, radial
  profiles, the radio-Sun size, and N/E/S/W sector profiles with the equator–pole
  elongation (the main measure for 2.5D axisymmetric runs).

The manual (Chapter "Comparing PLUTO output with observations") gives the
step-by-step recipe. It also explains what a 2.5D run can and cannot be compared with.

## Other stars

```python
from radiosynth import stellar as st
grid = rs.load_mhd("star_box.dat", r_star=0.42)          # coordinates in R_star
img = st.make_stellar_image(grid, "1.4GHz, 5GHz, 15GHz", inclination_deg=70, phase_deg=0,
                            distance_pc=10, fov=12, npix=160, gaunt="dulk")
img.flux_density_mjy                                     # mJy at 10 pc
sp = st.stellar_spectrum(grid, np.geomspace(0.5e9, 50e9, 12), distance_pc=10, gaunt="dulk")
lc = st.rotational_modulation(grid, "5GHz", range(0, 360, 20), inclination_deg=70, distance_pc=10)
```

* `r_star` sets the length unit to R_star; the coordinates are then in stellar radii.
* `distance_pc` switches to parallel rays and gives the flux at that distance.
* Images can be plotted in milliarcseconds with `units="mas"`.
* The command-line options are `--r-star`, `--distance-pc`, `--inclination` and `--phase`.
* Validation: the optically thick wind n ∝ r⁻² (S ∝ ν^(2/3)) is reproduced to 0.02 %
  in flux and 0.08 % in optical depth per pixel.
* Only thermal free-free emission is included. Gyrosynchrotron and coherent emission are not.

## Input notes

**BATS-R-US.** The density is taken from `Rho` (in g/cm³ for dimensional
output; other units are detected from the variable name or the IDL header,
and otherwise guessed from the magnitude). As in BATS-R-US, n_e = ρ/m_p. The
electron temperature is taken from the first of these that exists:

1. `Te`
2. `Pe/(n_e k)`
3. `P/(2 n_e k)`

If the points are not on a tensor grid (AMR `3d` output), they are resampled
to a uniform box. Use `--resample N` and `--resample-box`. It is better to
write a uniform `box` plot from BATS-R-US in the first place.

**PLUTO.** Supported formats:

* `.dbl` or `.flt`, as single_file or multiple_files, little- or big-endian;
* legacy VTK.

Supported geometries:

* `CARTESIAN`, which gives an `MHDGrid`;
* `SPHERICAL`, which gives a `SphericalGrid` and is sampled natively in
  (r, θ, φ), so no resampling is needed. This covers full 3D runs and 2.5D
  axisymmetric (r, θ) runs.

Reading through **pyPLUTO** is also supported: `load_pluto(dir, reader="pypluto")` or
`--pypluto` on the command line. The classic pyPLUTO is bundled in
`radiosynth/contrib/pyPLUTO.py`, with its `multiple_files` bug fixed. You can point to
your own copy with `pypluto_path=`. The fields agree with the built-in reader.

Code units are converted with `UNIT_DENSITY`, `UNIT_LENGTH` and
`UNIT_VELOCITY` from `definitions.h`. Expressions that use `CONST_*` names
are evaluated. You can also pass the units as `--unit-*` flags or keywords.
PLUTO's own default length unit is 1 AU, and the reader warns you if the inner
radius doesn't look like ~1 R☉.

The plasma is converted as follows:

* n_e = ρ/(μ_e m_p), where `mu_e` = 1 by default (BATS-R-US convention).
  Use about 1.17 for H+He.
* T = p μ m_p/(ρ k), where `mean_mol_weight` μ = 0.5 by default, i.e. T_e = T_i.
* A `T`/`Te` variable in the file takes precedence over this. For isothermal
  runs, pass `te_const`.

## Ray-tracing algorithm in detail

The image geometry is the same as in `write_plot_radiowave`:

* the image plane passes through the Sun's centre;
* `Tau = z × n` and `Xi = n × Tau`;
* the pixel centres are at the same positions;
* `rIntegration = ceil(max(diag+1, 5))`.

Each ray step is a line-by-line port of `ray_path`:

* predictor half step;
* `check_step` (curvature tolerance 10⁻², steepness test and distance to the
  critical surface);
* Boris step or parabola switch;
* `get_new_step_size` regrowth towards the local cell size.

The default mode differs from the Fortran in two places. Both follow the
algorithm as described in Benkevitch+2010 §5. You can get exact Fortran
behaviour with `strict_batsrus=True`.

1. **No spurious parabola switches.** The Fortran also switches when
   `ds < StepMin .and. ∇ε·v < 0`. That condition is reached only after the
   curvature control has shrunk the step, so it fires on rays far from the
   critical surface (ε ≈ 0.6). Those rays then jump along a long, wrong
   parabola. On an exact analytic field this put the turning point up to
   0.1 R☉ too high and the exit direction up to 10° off. Without it, the
   errors are 2×10⁻⁵ R☉ and 0.03°.
2. **Rays that cross the critical surface are kept.** The Fortran discards a
   ray that crosses the critical surface. Here the ray is stepped back, and as
   a last resort reflected by Snell's law, as described in both papers. In the
   strict mode, head-on rays are lost at coarse steps, so the flux depends on
   the step size.

Three smaller additions:

* the last segment before the photosphere is clipped at r_body;
* rays begin where they enter the model domain, which matters when the MHD box
  is smaller than the integration sphere;
* the vectorised NumPy loop runs over all rays at once, with optional
  `multiprocessing` (`n_workers`).

### Intensity

`transfer="thin"` (the default) gives I = Σ κ B ds. This is what BATS-R-US
does and what Moschou+2018 Eq. 14 uses. It is optically thin, so in the dense
low corona T_b can exceed T_e.

`transfer="full"` solves dI/ds = κ(B − I) and returns T_b ≤ T_e. It also
stores τ for each pixel. Use it for physically meaningful brightness
temperatures below about 1 GHz.

Other options:

* `gaunt="dulk"` uses the Coulomb-logarithm Gaunt factor;
* `kappa_over_mu=True` adds the 1/μ enhancement of absorption near the
  critical layer.

The image object provides:

* `img.brightness_temperature`, the Rayleigh–Jeans T_b in K;
* `img.flux_density`, in W m⁻² Hz⁻¹ at the observer;
* `img.tau`;
* `img.status`, which records why each ray ended: escaped, hit body,
  over-dense, and so on.

## Validation (`tests/`)

| test | result |
|---|---|
| Bouguer invariant μ\|r×v\| along rays, 80 MHz, gridded Newkirk corona | 1.2 % max deviation |
| turning radius vs analytic μ(r₀)r₀ = b | 4×10⁻⁴ R☉ |
| exit direction vs independent `solve_ivp` solution of the ray equation, 150 MHz | ≤ 0.3° (0.03° on the exact field) |
| straight-ray intensity vs quadrature, 1 and 17 GHz | ≤ 0.9 % |
| full transfer: T_b ≤ T_e | yes |
| flux vs step size (step_factor 1 → 0.25) | 3×10⁻⁴ |
| BATS-R-US Tecplot ASCII / IDL real4 / IDL ascii / scattered-point readers | exact / 3×10⁻³ |
| PLUTO dbl (3D spherical), flt (2.5D axisymmetric), VTK vs Cartesian reference images | flux within 1–3.5 %, log-image correlation 0.9999 |
| pyPLUTO reader vs built-in reader | identical (≤2.5×10⁻⁶, float32) |
| gyroresonance layer τ vs Maxwellian ⟨J²ₛ₋₁⟩ (exact Bessel), s = 2–6 | ≤ 0.8 % |
| single s = 3 layer: I = S(1−e^−τ) per mode; thick layers T_b = T_e | 10⁻⁶; 2×10⁻¹¹ |
| V sign over positive spot, V → −V for B → −B; thin free-free V/I = 2q/(1+q²) | exact; 3×10⁻⁴ |
| straight-ray free-free vs refractive tracer without refraction | 5×10⁻⁴ |
| spherical (B_r,B_θ,B_φ) vs Cartesian; PLUTO bx1..3 → gauss | 0.8–1.8 %; 10⁻¹¹ |
| compare_refraction vs separate runs; 17 GHz refracted/straight flux | identical; 1 − 2×10⁻⁴ |
| combined without fast electrons vs gyroresonance image | identical |
| combined gyrosynchrotron-only (per-voxel coefficients) vs library transfer | 10⁻¹⁵ |
| combined thin limit vs sum of parts; thick layer hides GS source | 8×10⁻⁶; T_b = T_e |
| combined ff_refraction (free-free only) vs refracted make_radio_image | ≤ 1.1×10⁻³ |
| 1D symmetric image vs full 2D image (flux; T_b along a row) | 2×10⁻⁴; 3×10⁻⁴ |
| 1D optically thick wind from a text table vs exact integral | 2×10⁻⁴ |
| gyrosynchrotron wrapper vs direct library call; thick thermal slab T_b = n²T_e | 3×10⁻⁷; exact |
| thin GS emissivity vs Dulk & Marsh (1982) approximation | ratio 1.3–2.0 |
| array-defined vs analytical power law; GS V sign; library vs radiosynth thin free-free | 3×10⁻³; exact; +15 % (I), ≤3 % (V/I) |

To run them:

```bash
python tests/validate.py
python tests/test_readers.py
python tests/test_pluto.py
python tests/test_orientation.py
python tests/test_stellar.py
python tests/test_gyroresonance.py
python tests/test_gyrosynchrotron.py     # needs the GS codes (skipped otherwise)
python tests/test_refraction_compare.py
python tests/test_combined.py            # gyrosynchrotron parts need the GS codes
python tests/test_symmetric.py
python examples/demo.py
python examples/moschou_figures.py
python examples/gyroresonance_example.py
python examples/gyrosynchrotron_example.py
python examples/combined_example.py
python examples/symmetric_example.py
```

Documentation (two PDFs in `doc/`):

* `radiosynth_user_manual.pdf` — user guide (input, imaging, mechanisms,
  gyroresonance, figures, comparison with observations, stars) and theory;
* `radiosynth_code_reference.pdf` — line-by-line description of the source,
  folder by folder.

Rebuild with `cd doc; python gen_coderef.py; xelatex radiosynth_user_manual.tex`
(three times) and then `xelatex radiosynth_code_reference.tex` (three times).

## Caveats

* Plasma emission and scattering are not included. As Moschou+2018 note,
  scattering matters below about 200 MHz.
* Refraction is followed for free-free and simplistic emission only; gyroresonance
  and gyrosynchrotron images use straight lines of sight.
* The model has to resolve the density scale height near the critical layer.
  At about 0.025–0.05 R☉ cells, the accuracy is around 1 %.
* High-frequency images (≳1 GHz) need the chromosphere and transition region
  in the model. Above 1 GHz the synthetic corona used in the tests is mostly
  optically thin, so the disk looks dark.

## Licence

radiosynth is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version. It is distributed in
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [`LICENSE`](LICENSE).

`radiosynth/common/contrib/pyPLUTO.py` is the pyPLUTO reader distributed with the PLUTO
code (PLUTO team), included with a small fix. The fast gyrosynchrotron codes
(GPL-3.0) are not included; they are downloaded and built by the user.

## Citing radiosynth

If you use radiosynth in a publication, please cite the software (see
[`CITATION.cff`](CITATION.cff)) and the papers its algorithms are based on:

* Benkevitch, L., Sokolov, I., Oberoi, D., & Zurbuchen, T. 2010, ApJ 725, 447 — ray-tracing algorithm;
* Moschou, S.-P., Sokolov, I., Cohen, O., et al. 2018, ApJ 867, 51 — bremsstrahlung images from MHD models;
* Kuznetsov, A. A., & Fleishman, G. D. 2021, ApJ 922, 103, and Fleishman, G. D., & Kuznetsov, A. A. 2010, ApJ 721, 1127 — fast gyrosynchrotron codes (for gyrosynchrotron and combined images).

## Author and acknowledgements

radiosynth is written by **Soumitra Hazra**.

The code follows previously existing codes and published algorithms: the radio-wave
imaging module of BATS-R-US in the Space Weather Modeling Framework
(`ModRadioWaveImage.f90`, `ModWritePlotRadioWave.f90`, `ModUserAwsom.f90`; Benkevitch et
al. 2010; Moschou et al. 2018), the pyPLUTO reader of the PLUTO code, and the fast
gyrosynchrotron codes of Fleishman & Kuznetsov, which it calls as a library. The
gyroresonance and magnetised free–free formulas follow the literature cited in the
User Manual. radiosynth was developed with the help of large language models. Its results are
checked against analytic solutions and independent calculations by the test suite in
`tests/` (see *Validation* above).

Bug reports, questions and contributions are welcome through the issue tracker of the
repository.

