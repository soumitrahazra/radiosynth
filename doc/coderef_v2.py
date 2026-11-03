# -*- coding: utf-8 -*-
# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Commentary of the code reference, grouped by folder (common/, simplistic/,
bremsstrahlung/, gyroresonance/, gyrosynchrotron/, combined/).

Starts from the base commentary (coderef_content.MODULES), places it at the package
paths, adds the remaining modules and fixes the order.
"""
from coderef_content import MODULES as _OLD

M = {m[0]: list(m) for m in _OLD}


def entry(path, title, intro, blocks):
    M[path] = [path, title, intro, blocks]


def insert_block(path, after_heading, block):
    blocks = M[path][3]
    for i, b in enumerate(blocks):
        if b[1] == after_heading:
            blocks.insert(i + 1, block)
            return
    raise KeyError(after_heading)


def drop_blocks(path, headings):
    M[path][3] = [b for b in M[path][3] if b[1] not in headings]


def move(old, new, title):
    m = M.pop(old)
    m[0] = new
    m[1] = title
    M[new] = m


# ---------------------------------------------------------------------------
# common/: existing modules moved
# ---------------------------------------------------------------------------
for name in ("constants", "grid", "raytrace", "image", "readers", "pluto", "figures",
             "fitsio", "stellar"):
    t = M[f"{name}.py"][1]
    move(f"{name}.py", f"common/{name}.py", "common/" + t)

# --- grid: magnetic field
insert_block("common/grid.py", "MHDGrid.sample(): trilinear interpolation", (
    "# ---- magnetic field (gyroresonance", "MHDGrid: magnetic field", [
        "The field is optional and stored separately from $n_e$ and $T_e$, so grids without a field use no extra memory and the refractive tracer is unchanged.",
        ("def has_field", "True once a field has been attached."),
        (("def add_field(self, bx, by, bz", "self._bflat = b.astype"), "Stack $(B_x,B_y,B_z)$ [G] on the grid nodes, check the shape, replace non-finite values by 0 and keep them in `float32` as a (N,3) array."),
        (("def _interp_flat", "return out"), "Generic trilinear interpolation of any (nodes, k) array, with the same cell location and the same eight weights as `sample()`."),
        ("def sample_field", "$\\vect B$ [G] at the points, Cartesian components."),
        (("def add_scalar", "return self"), "Any named scalar field on the nodes (stored in `grid.scalars`), e.g. the current density `'j'` (common/fields.py), for use in fast-electron rules."),
        ("def sample_scalar", "Its value at the points."),
    ]))
insert_block("common/grid.py", "SphericalGrid.sample(): gradient to Cartesian, cell size", (
    "# ---- magnetic field (gyroresonance", "SphericalGrid: magnetic field", [
        (("def add_field(self, b1, b2, b3", "self._bflat = b.astype"), "The field is given at the cell centres either as spherical components $(B_r,B_\\theta,B_\\phi)$ --- PLUTO's `bx1`, `bx2`, `bx3` --- or as Cartesian components. The choice is remembered in `_bcomp`."),
        (("def _interp_flat", "return out, th, ph"), "Generic interpolation in $(r,\\theta,\\phi)$ index space of any (cells, k) array, with the same longitude handling as `sample()` (axisymmetric, periodic, or a longitude patch); also returns $\\theta$, $\\phi$ of the points."),
        (("def sample_field", "return out"), "$\\vect B$ at the points; Cartesian components are returned as they are."),
        (("br, bt, bp = out.T", "return b"), "Spherical to Cartesian at the sample point: $B_x=B_r\\sin\\theta\\cos\\phi+B_\\theta\\cos\\theta\\cos\\phi-B_\\phi\\sin\\phi$, and similarly for $B_y$, $B_z$. Interpolating the spherical components and rotating afterwards keeps a radial field radial between grid points."),
        (("def add_scalar", "def sample_scalar"), "Named scalar fields at the cell centres, as for the Cartesian grid."),
    ]))

insert_block("common/image.py", "trace_paths(): individual trajectories", (
    "REFRACTIVE_MECHANISMS = (", "compare_refraction(): refracted vs straight lines of sight", [
        ("REFRACTIVE_MECHANISMS = (", "The mechanisms that use the refractive tracer; gyroresonance and gyrosynchrotron are straight-ray only and are refused."),
        (("common = dict(", "los = make_radio_image"), "The same call twice; only `no_refraction` differs, so the comparison isolates refraction."),
        ("flux_ratio = ", "Refracted / straight flux density per frequency."),
        (("with np.errstate(invalid=\"ignore\", divide=\"ignore\"):", "changed = np.array"), "Fraction of the emitting pixels (above $10^{-3}$ of the maximum) whose intensity changes by more than 10\\%."),
        (("if verbose:", "fig = _figure_refraction"), "Report and optional figure."),
        (("def _figure_refraction", "return fig"), "One row per frequency: refracted and straight maps on a common logarithmic scale, and $\\log_{10}$ of their ratio."),
    ]))

# --- readers: magnetic field
insert_block("common/readers.py", "Unit tables", (
    "_B_UNITS = {", "find_magnetic_field(): Bx, By, Bz in gauss", [
        ("_B_UNITS = {", "Gauss per unit: G = 1, nT = $10^{-5}$, T = $10^4$, mT = 10."),
        (("for cand in ((\"bx\", \"by\", \"bz\")", "break"), "Look for `Bx, By, Bz`, then `B_x, ...`, then `b1, b2, b3` (names compared without units and case)."),
        (("if b_unit is not None", "src = f\"b_unit"), "An explicit `b_unit` wins: a number (gauss per file unit) or a unit name."),
        (("u = _unit(names[0])", "warnings.warn"), "Otherwise the unit in the variable name (e.g. `Bx [Gauss]`, BATS-R-US Tecplot) is used; without one, gauss is assumed with a warning. BATS-R-US IDL files in normalised units need `b_unit`."),
    ]))

for _b in M["common/pluto.py"][3]:
    if _b[1] == "load_pluto(): grid":
        _b[2].append((("one_d = ", "assuming spherical symmetry\")"),
                      "1D spherical output (one $\\theta$ and one $\\phi$ cell) is read as a "
                      "spherically symmetric model with `grid_from_profile` (common/symmetric.py)."))
# --- pluto: with_b
insert_block("common/pluto.py", "load_pluto(): grid", (
    "    if with_b:", "load_pluto(): magnetic field", [
        ("b = [_find_var(v, f\"bx{i}\")", "Cell-centred `bx1`, `bx2`, `bx3` (the staggered `bx1s`... are not used)."),
        ("bfac = math.sqrt", "PLUTO code units to gauss: $B=b\\sqrt{4\\pi\\rho_0}\\,v_0$ with $\\rho_0$ = `UNIT_DENSITY` and $v_0$ = `UNIT_VELOCITY` (cgs)."),
        (("if geom == \"SPHERICAL\":", "g.add_field(*b)"), "Spherical runs store $(B_r,B_\\theta,B_\\phi)$; VTK files written with `VTK_VECTOR_DUMP` hold Cartesian vectors (`b_cartesian`). Cartesian runs store $(B_x,B_y,B_z)$."),
    ]))

# --- stellar: the wind solution moved to bremsstrahlung/wind.py
_wind = [b for b in M["common/stellar.py"][3] if b[1].startswith("wind_")]
drop_blocks("common/stellar.py", [b[1] for b in _wind])
M["common/stellar.py"][3].append((
    "# The analytic free-free wind solution", "Re-export of the wind solution", [
        ("from ..bremsstrahlung.wind import", "The analytic wind functions now live in `bremsstrahlung/wind.py` (they are free--free physics); they are imported here so that `radiosynth.stellar.wind_flux_density` keeps working."),
    ]))
M["common/stellar.py"][2] = M["common/stellar.py"][2].replace(
    ", and the analytic optically thick wind solution\nused for validation", "")

# --- emission.py: now only the dispatcher; physics in the mechanism folders
_old_em = M.pop("emission.py")
entry("common/emission.py", "common/emission.py --- the emission-model interface",
r"""`EmissionModel` is the callable object used by the refractive ray tracer: it
returns the absorption coefficient $\kappa$ [m$^{-1}$] and the source function $S$ for
arrays of $n_e$ and $T_e$ at one frequency (the Python counterpart of
`user_material_properties` in `ModUserAwsom.f90`). It holds no physics itself:
it looks the mechanism up in the registry and forwards the call to
`simplistic/emission.py` or `bremsstrahlung/emission.py`. You can still pass any
function with the same signature as `emission=`.""",
[
('"""Emission-model interface', "Module docstring and imports", [
    ("from .mechanism import get_mechanism", "The registry (common/mechanism.py) is the only link to the mechanism folders."),
]),
("class EmissionModel", "Constructor", [
    (("kind = kind.lower()", "mech = get_mechanism(kind)"), "Case-insensitive lookup; an unknown name raises an error listing the available mechanisms (as `CON_stop` does in Fortran)."),
    (("if mech.coefficients is None", "raise ValueError"), "Gyroresonance needs the magnetic field and has its own driver, so it cannot be used here."),
    ("self._coef = mech.coefficients", "Keep the coefficient function."),
    ("def __repr__", "Readable description, printed by `make_radio_image`."),
]),
("def __call__", "Evaluation: __call__(ne, te, freq, length_unit)", [
    ("if self.kind == \"bremsstrahlung\"", "Bremsstrahlung also receives the Gaunt factor."),
    ("return self._coef(np.asarray", "Other mechanisms get $n_e$, $T_e$, the frequency and the length unit."),
]),
])

_brems_items = [b for b in _old_em[3] if b[1] == "Evaluation: __call__(ne, te, freq)"][0][2]
entry("simplistic/emission.py", "simplistic/emission.py --- the simplistic mechanism",
r"""The ad-hoc emissivity of Benkevitch et al. (2010), kept in BATS-R-US.""",
[
('"""The \'simplistic\'', "Module docstring and imports", [
    "States the formula and what the mechanism is for: a dimensionless map of the layer near the critical density, useful to see refraction.",
]),
("def simplistic_coefficients", "simplistic_coefficients()", [
    ("r = ne / critical_density", "$r=n_e/n_\\text{cr}$."),
    ("kappa = r**2 * (0.5 - r)**2 / length_unit", "$\\kappa=r^2(0.5-r)^2$ per normalised length. Dividing by `length_unit` ($R_\\odot$, or $R_\\ast$ for a star) gives a per-metre coefficient, so that $\\kappa\\,\\Delta\\ell$[m] equals the Fortran product of $\\kappa$ and the normalised step."),
    ("return kappa, np.ones_like(ne)", "$S=1$, a proxy as in Fortran."),
]),
])
entry("simplistic/__init__.py", "simplistic/__init__.py --- package interface",
r"""Registers the mechanism and provides `make_image`.""",
[
('"""Simplistic mechanism', "Mechanism registration", [
    ("def make_image", "`make_radio_image` with `emission='simplistic'`; every other keyword is passed through."),
    ("def compare_refraction", "Refracted vs straight-line images with the simplistic emissivity (common/image.py `compare_refraction`)."),
    ("MECHANISM = register(Mechanism(", "Entry in the registry: name, coefficient function, image driver and a one-line description (shown by `--list-mechanisms`)."),
]),
])

entry("bremsstrahlung/emission.py", "bremsstrahlung/emission.py --- thermal free--free coefficients",
r"""The bremsstrahlung coefficients of `ModUserAwsom.f90` (Moschou et al. 2018), or of
Dulk (1985).""",
[
('"""Thermal free-free', "Module docstring and imports", [
    "The formulas are written out in the User Manual, Part II.",
]),
("_BREMS_PREF", "The prefactor", [
    ("_BREMS_PREF", "Pre-computes the constant part of $\\kappa$: $\\frac43\\sqrt{2\\pi/3}\\,(e^2/4\\pi\\varepsilon_0)^3/c$. This is the Fortran expression `4/3*sqrt(2*cPi/3)*(cElectronChargeSquaredJm/ ...)**3 /cLightSpeed` with the temperature factor taken out."),
]),
("def bremsstrahlung_coefficients", "bremsstrahlung_coefficients()", [
    ("ne = np.asarray", "Accept lists or arrays; all arithmetic in double precision."),
] + [it for it in _brems_items if isinstance(it, tuple) and isinstance(it[0], (str, tuple))
     and (it[0] if isinstance(it[0], str) else it[0][0]).startswith(("te = np", "B = 2.0", "if isinstance(self.gaunt", "kappa = (_BREMS"))]),
])
# adapt markers that changed name (B -> S, self.gaunt -> gaunt)
_b = M["bremsstrahlung/emission.py"][3][2][2]
for i, it in enumerate(_b):
    ref, txt = it if isinstance(it, tuple) else (None, it)
    if ref == "B = 2.0 * freq":
        _b[i] = ("S = 2.0 * freq", txt)
    elif isinstance(ref, tuple) and ref[0] == "if isinstance(self.gaunt, str)":
        _b[i] = (("if isinstance(gaunt, str)", ref[1]), txt)
_b.append(("return kappa, S", "Both arrays have the shape of the input. `length_unit` is accepted only so that all mechanisms share one signature."))

entry("bremsstrahlung/magnetized.py", "bremsstrahlung/magnetized.py --- free--free emission of the two magneto-ionic modes",
r"""Free--free absorption coefficients of the o- and x-modes in a magnetic field, used by
the polarised straight-ray transfer of the gyroresonance package (for
`mechanisms='gr+ff'` and `'ff'`).""",
[
('"""Free-free absorption', "Module docstring and imports", [
    "Gives the approximation $\\kappa_\\sigma=\\kappa_\\text{ff}/(1+\\sigma Y|\\cos\\theta|)^2$ ($\\sigma=+1$ o-mode, $-1$ x-mode), Kirchhoff's law per mode with $S_\\sigma=B_\\nu/2$, and the resulting thin-source polarisation $P=2Y|\\cos\\theta|$.",
]),
("def magnetized_ff_coefficients", "magnetized_ff_coefficients()", [
    ("kappa, S = bremsstrahlung_coefficients", "The unmagnetised coefficient (Gaunt factor 10 or Dulk) and $B_\\nu$."),
    ("_, Y = magnetoionic_params", "$Y=\\nu_B/\\nu$ from $|B|$."),
    ("q = np.minimum", "$q=Y|\\cos\\theta|$, capped at `y_max` = 0.9: the expansion is only valid for $Y<1$, i.e.\\ away from the $s=1$ layer, where gyroresonance dominates anyway."),
    (("kappa_o = kappa / (1.0 + q) ** 2", "kappa_x = kappa / (1.0 - q) ** 2"), "o-mode less, x-mode more absorbing. For $q\\ll1$, $(\\kappa_x-\\kappa_o)/(\\kappa_x+\\kappa_o)=2q/(1+q^2)$ --- tested in `tests/test_gyroresonance.py`."),
    ("S_mode = 0.5 * S", "Each mode carries half of the Rayleigh--Jeans intensity, so $I_o+I_x=B_\\nu$ when thick."),
]),
("def rayleigh_jeans_mode", "rayleigh_jeans_mode()", [
    "$\\nu^2k_BT/c^2$, the source function of one mode.",
]),
])

entry("bremsstrahlung/wind.py", "bremsstrahlung/wind.py --- analytic stellar-wind solution",
r"""The analytic free--free optical depth and flux of an isothermal wind
$n=n_0(r_0/r)^2$, used to validate the ray tracer for stars (also available as
`radiosynth.stellar.wind_*`).""",
[('"""Analytic reference', "Module docstring and imports", [
    "The functions use `EmissionModel('bremsstrahlung')` to get $\\kappa/n_e^2$, so they share the Gaunt-factor options of the images.",
])] + _wind)

entry("bremsstrahlung/__init__.py", "bremsstrahlung/__init__.py --- package interface",
r"""Registers thermal free--free emission and collects its functions.""",
[('"""Thermal free-free', "Mechanism registration", [
    (("from .emission import", "from .wind import"), "Coefficients, two-mode coefficients, wind solution."),
    ("def make_image", "`make_radio_image` with `emission='bremsstrahlung'` (refractive ray tracing; `gaunt`, `transfer` and all other keywords passed through)."),
    ("def compare_refraction", "Refracted vs straight-line free--free images (common/image.py `compare_refraction`)."),
    ("MECHANISM = register(Mechanism(", "Registry entry; `options` lists the mechanism-specific keyword (`gaunt`)."),
])])

# ---------------------------------------------------------------------------
# common/: new modules
# ---------------------------------------------------------------------------
entry("common/mechanism.py", "common/mechanism.py --- the mechanism registry",
r"""The single place where `common/` meets the mechanism folders. A new mechanism is a
new folder that calls `register()` --- nothing in `common/` changes.""",
[
('"""Registry of radio', "Module docstring", [
    "Describes the fields of an entry: `name`, `coefficients` (for the refractive tracer, or None), `make_image`, `description`.",
]),
("import importlib", "Registry storage", [
    ("_REGISTRY = {}", "Name $\\to$ `Mechanism`."),
    ("BUILTIN = (", "The folders shipped with radiosynth; they are imported on demand, which avoids circular imports between `common/` and the mechanisms."),
]),
("class Mechanism", "class Mechanism", [
    ("def __init__", "A plain record; `options` names the mechanism-specific keywords (for documentation and the CLI)."),
]),
("def register", "register()", [
    "Adds or replaces an entry (lower-case name) and returns it, so that a package can write `MECHANISM = register(Mechanism(...))`.",
]),
("def get_mechanism", "get_mechanism()", [
    (("if key not in _REGISTRY and key in BUILTIN", "importlib.import_module"), "Import `radiosynth.<name>` the first time a built-in mechanism is asked for."),
    (("if key not in _REGISTRY:", "raise ValueError"), "Unknown names raise an error listing the available ones."),
]),
("def available_mechanisms", "available_mechanisms()", [
    "Imports all built-in folders and returns the sorted names (`python -m radiosynth --list-mechanisms`).",
]),
])

entry("common/magnetoionic.py", "common/magnetoionic.py --- cold-plasma magneto-ionic theory",
r"""Shared by the magnetised free--free coefficients and the gyroresonance opacity: the
parameters $X$, $Y$, the axial ratio of the modes, the Appleton--Hartree refractive
index and the cutoffs.""",
[
('"""Cold-plasma', "Module docstring and imports", [
    "Defines the notation ($X$, $Y$, $\\theta$ with $\\vect k$ towards the observer), the modes, the Appleton--Hartree formula, the polarisation vector $(T\\cos\\theta,\\,i,\\,-T\\sin\\theta)/\\sqrt{1+T^2}$ and the handedness convention: for $\\vect B\\cdot\\vect k>0$ the x-mode is right-hand circularly polarised (IAU/IEEE), $V>0$.",
]),
("NU_B_PER_GAUSS", "Gyrofrequency", [
    ("NU_B_PER_GAUSS", "$\\nu_B=eB/(2\\pi m_e)$ per gauss: 2.7992 MHz/G."),
]),
("def gyrofrequency", "gyrofrequency() and magnetoionic_params()", [
    ("X = np.asarray(ne", "$X=n_e/n_\\text{cr}=(\\nu_p/\\nu)^2$ (same $n_\\text{cr}$ as the ray tracer)."),
    ("Y = gyrofrequency", "$Y=\\nu_B/\\nu$."),
]),
("def axial_ratio_x", "axial_ratio_x()", [
    ("den = Y * s2 + np.sqrt", "$|T_x|=2(1-X)|\\cos\\theta|/[Y\\sin^2\\theta+\\sqrt{Y^2\\sin^4\\theta+4(1-X)^2\\cos^2\\theta}]$: 1 (circular) along $\\vect B$, 0 (linear, $\\vect E\\perp\\vect B$) across it."),
    ("return np.where(den > 0", "Guard against $0/0$ when $Y=0$ and $\\cos\\theta=0$."),
]),
("def refractive_index2", "refractive_index2()", [
    (("root = np.sqrt", "den = a - 0.5"), "Appleton--Hartree with the $+$ root for the o-mode and $-$ for the x-mode: along $\\vect B$, $n^2=1-X/(1\\pm Y)$."),
]),
("def cutoff", "cutoff()", [
    "o-mode: $X\\ge1$; x-mode: $X\\ge1-Y$. The transfer stops a polarisation at its cutoff.",
]),
])

entry("common/straight.py", "common/straight.py --- straight-ray sampler",
r"""Cuts each line of sight into equal steps and samples $n_e$, $T_e$ and $\vect B$ at the
nodes, chunk by chunk, for mechanisms that integrate along straight rays.""",
[
('"""Straight-ray sampler', "Module docstring and imports", [
    "Explains why straight rays suffice for gyroresonance (the layers form where $\\nu\\gg\\nu_p$) and the conventions: same start points as the refractive tracer, node 0 nearest the observer.",
]),
("def segment_ends", "segment_ends(): where each line ends", [
    ("_, t_out = grid.domain_intersection", "Exit from the grid (box or outer sphere)."),
    (("disc = b * b - (c - r_integration", "t_end = np.minimum"), "Far exit of the integration sphere, $|\\vect p+t\\vect d|=r_\\text{int}$."),
    (("disc_b = b * b - (c - grid.r_body", "t_end = np.where(hit"), "First intersection with the body sphere $r=r_\\text{body}$, if it comes before the exit."),
    ("status = np.where", "Status as in the tracer: escaped, hit body or missed grid."),
]),
("def default_step", "default_step()", [
    "Half the smallest cell, but never more than 20000 nodes per ray. The harmonic layers are located between nodes, so the step only has to resolve the variation of the fields.",
]),
("def iter_samples", "iter_samples(): chunks of sampled rays", [
    (("nk_all = np.ceil", "order = np.argsort"), "Number of steps per ray; rays are sorted by length so that each chunk is nearly rectangular."),
    (("nk = max(int(nk_all[order[start]])", "nr = stop - start"), "Chunk size: as many rays as fit in `max_points` nodes for the longest ray of the chunk (memory stays bounded)."),
    (("t = np.minimum(k[None, :] * ds", "p = pos[idx]"), "Node parameters (the last step of a ray is shorter, later steps have zero length) and positions."),
    (("ne, _, te, _, ins = grid.sample", "dl=np.diff"), "Sample $n_e$ and $T_e$ (zero density outside the domain) and the step lengths."),
    (("if with_field:", "out[\"b\"] ="), "Sample $\\vect B$ [G], zero outside the domain."),
]),
])

# --- cli and package files
move("__main__.py", "common/cli.py", "common/cli.py --- command-line interface")
entry("common/cli.py", "common/cli.py --- command-line interface",
r"""`python -m radiosynth INPUT --freq ...`: parses the options, loads the model with
`load_mhd` (with the magnetic field for gyroresonance), calls the image driver of the
chosen mechanism, and writes the requested files.""",
[
('"""Command-line', "Module docstring and imports", [
    "Two examples: a free--free run and a gyroresonance run.",
    ("from .mechanism import", "For `--list-mechanisms`."),
]),
("def main(argv=None)", "Options: image and physics", [
    ("ap.add_argument(\"input\"", "Model file or PLUTO directory (optional only so that `--list-mechanisms` works alone)."),
    (("--freq", "--up"), "Frequencies, observer (or `--date`, `--frame`, `--phi-offset` to place it at the Earth for a given time), stellar options, image size, pixels and up vector."),
    (("--mechanism", "--gaunt"), "Mechanism (`--emission` is kept as an alias) and Gaunt factor (default 10, or 'dulk' for gyroresonance)."),
    (("gr = ap.add_argument_group", "--b-unit"), "Gyroresonance options: which contributions, highest harmonic, mode coupling, step, body temperature, unit of $B$."),
    (("gs = ap.add_argument_group", "--gs-diagnostics"), "Gyrosynchrotron options: fast-electron density and its spatial rule, energy spectrum, pitch-angle distribution, mode coupling, library path, diagnostics."),
    (("--ff-refraction", "--include"), "Combined image: refraction correction of the free--free part and the processes to include."),
    (("--transfer", "--workers"), "Transfer mode, refraction switches (`--no-refraction`, `--compare-refraction`) and numerical controls of the refractive tracer."),
    (("--density-var", "--save-grid"), "Input interpretation for BATS-R-US and generic files, and resampling."),
]),
("sy = ap.add_argument_group", "Options: 1D models", [
    (("sy = ap.add_argument_group", "--n-p"), "Text-table input of a 1D model (columns, density kind and unit, radius unit), and `--symmetric` (one row of rays) with its number of rays."),
]),
("pl = ap.add_argument_group", "Options: PLUTO and output", [
    (("--nout", "--mmw"), "PLUTO output number, format, definitions file, units and plasma parameters."),
    (("--out", "--plot"), "Output prefix, file formats and plots."),
]),
("a = ap.parse_args", "Running", [
    (("if a.list_mechanisms", "return 0"), "List the registered mechanisms and stop."),
    ("use_gr = ", "Gyroresonance and gyrosynchrotron need the field and their own drivers."),
    (("if a.compare_refraction and", "use straight rays)"), "`--compare-refraction` is refused for the straight-ray mechanisms."),
    (("rs = None", "box = ((b[0]"), "Convert the resampling options to tuples."),
    (("if a.profile:", "grid._flat[:, 4] = a.te_const"), "A 1D text table is read with `read_profile` (free-free and simplistic only); without a temperature column `--te-const` sets $T_e$."),
    ("grid = load_mhd(", "Load the model; `with_b` and `b_unit` only for gyroresonance."),
    ("grid.to_npz", "Optionally cache the grid (with $\\vect B$ if loaded)."),
]),
("g_ = a.gaunt", "Imaging and output", [
    (("if a.date:", "observer = observer_for_date"), "With `--date`, the observer is computed by `observer_for_date`."),
    (("if a.inclination is not None", "observer = stellar_observer"), "With `--inclination`, the view direction comes from the inclination and phase."),
    (("if a.symmetric:", "flux density from the profile"), "`--symmetric`: the image from one row of rays, the radial profile written as text and figure, and the flux."),
    (("elif a.compare_refraction:", "return 0"), "`--compare-refraction`: both images with identical options, the comparison figure, and the files of both."),
    (("if use_gs or (use_cb", "rule = in_sphere"), "Gyrosynchrotron or combined: build the $n_b$ rule from the options (blob, uniform, fraction of $n_e$, or proportional to $|\\vect J|$ computed by `current_density`) and the `FastElectrons` prescription."),
    (("elif use_cb:", "tolerance=a.tolerance))"), "Combined image (with the optional refraction correction of the free--free part); nothing more to do after `--symmetric`."),
    (("elif use_gs:", "abundance="), "Gyrosynchrotron image."),
    (("elif use_gr:", "distance_pc=a.distance_pc)"), "Otherwise `make_gr_image` or `make_radio_image`."),
    (("if \"npz\" in a.formats", "img.plot(\"gs_fraction\""), "Write the requested files; for gyroresonance and gyrosynchrotron also Stokes V FITS files and polarisation maps, the harmonic map (gyroresonance) and the fast-electron fraction (gyrosynchrotron)."),
    (("if a.distance_pc:", "mJy"), "For a star, print the flux density at each frequency in mJy."),
]),
])

entry("__init__.py", "radiosynth/__init__.py --- package interface",
r"""Defines what `import radiosynth as rs` provides, and the short module paths
(`radiosynth.grid`, ...).""",
[
('"""radiosynth', "Docstring: the package layout", [
    "Describes the four folders and gives a free--free and a gyroresonance quick start.",
]),
("import sys as _sys", "Public names", [
    ("from .common import (", "The shared modules, also available as `rs.grid`, `rs.figures`, ..."),
    (("from .common.constants import", "from .common.mechanism import get_mechanism"), "The main classes and functions."),
    ("from . import simplistic, bremsstrahlung, gyroresonance, gyrosynchrotron, combined", "Importing the five folders registers their mechanisms (the gyrosynchrotron library itself is loaded only when an image is made)."),
    ("from .combined import make_combined_image", "The combined driver at top level."),
    (("from .gyroresonance import make_gr_image", "from .common.fields import current_density"), "The gyroresonance and gyrosynchrotron drivers, `FastElectrons` and `current_density` at top level."),
]),
("# short module paths", "Short module paths", [
    (("for _name, _mod in", "_sys.modules.setdefault(f\"{__name__}.contrib\""), "Registers the old module names (`radiosynth.grid`, `radiosynth.figures`, `radiosynth.contrib`, ...) as aliases of the modules in `common/`, so `from radiosynth.constants import R_SUN` and similar short imports work."),
    ("__version__", "Package version (2.2.0)."),
]),
])
entry("__main__.py", "radiosynth/__main__.py --- python -m radiosynth",
r"""Three lines: runs `common/cli.main`.""",
[('"""python -m radiosynth', "Entry point", [])])

# ---------------------------------------------------------------------------
# gyroresonance/
# ---------------------------------------------------------------------------
entry("gyroresonance/opacity.py", "gyroresonance/opacity.py --- optical depth of a harmonic layer",
r"""The thermal gyroresonance opacity in the thin-layer form used by gyroresonance codes
(Zheleznyakov 1970; White \& Kundu 1997; Fleishman \& Kuznetsov 2014).""",
[
('"""Thermal gyroresonance', "Module docstring and imports", [
    "Gives the formula, the definitions of $\\beta_T$, $L_B$ and the mode factor $F_\\sigma$, and the approximations (lowest order in $\\beta_T$, refractive index 1, longitudinal polarisation neglected). User Manual, Chapter~\\ref{chap:grtheory}, derives it.",
]),
("GR_PREFACTOR", "Prefactor", [
    ("GR_PREFACTOR", "$\\pi e^2/(m_ec)$ in SI, $\\pi(e^2/4\\pi\\varepsilon_0)/(m_ec)=2.654\\times10^{-6}$ m$^2$ s$^{-1}$ (0.02654 cm$^2$ s$^{-1}$ in cgs)."),
]),
("def mode_factor", "mode_factor()", [
    ("tx = axial_ratio_x", "$|T_x|$ at the layer, where $Y=1/s$."),
    ("return (1.0 + tx * c) ** 2", "x-mode: $F_x=(1+|T_x||\\cos\\theta|)^2/(1+T_x^2)$; 2 along $\\vect B$, 1 across."),
    ("return (tx - c) ** 2", "o-mode with $T_o=-1/T_x$: $F_o=(|T_x|-|\\cos\\theta|)^2/(1+T_x^2)$; 0 along and (at this order) across $\\vect B$."),
]),
("def layer_optical_depth", "layer_optical_depth()", [
    ("beta2 = K_B * te", "$\\beta_T^2=k_BT_e/(m_ec^2)$."),
    ("sin2 = np.maximum", "$\\sin^2\\theta$."),
    ("F = mode_factor(X, 1.0 / s, c, mode)", "Mode factor with $Y=1/s$."),
    (("return (GR_PREFACTOR * ne", "* (s * s / math.factorial(s))"), "$\\tau_s^\\sigma=\\frac{\\pi e^2n_eL_B}{m_ec\\nu}\\,\\frac{s^2}{s!}\\left(\\frac{s^2\\beta_T^2\\sin^2\\theta}{2}\\right)^{s-1}F_\\sigma$."),
]),
("def harmonic_field", "harmonic_field()", [
    "$|B|=\\nu/(s\\,\\nu_B/\\text{G})$: the field strength of the $s$-th layer, e.g.\\ 595 G for $s=3$ at 5 GHz.",
]),
])

entry("gyroresonance/transfer.py", "gyroresonance/transfer.py --- polarised transfer along straight rays",
r"""Combines harmonic layers and two-mode free--free absorption along each sampled line
of sight and returns Stokes I and V and diagnostics.""",
[
('"""Polarised radiative', "Module docstring", [
    "Describes the step-by-step scheme, the treatment of mode coupling (weak: transfer of right/left circular polarisation; strong: transfer of the modes) and of cutoffs.",
]),
("from ..common.constants import", "Imports", [
    ("_SRC = K_B / C_LIGHT**2", "$S=\\nu^2k_BT/c^2$ per mode."),
]),
("def transfer_chunk", "transfer_chunk(): fields along the rays", [
    ("use_gr = ", "Which contributions are included (`mechanisms`)."),
    ("nobs = -dirs", "Unit vector towards the observer, $\\vect k$ direction."),
    (("bmag = np.sqrt", "cos = np.where"), "$|B|$ and $\\cos\\theta=\\vect B\\cdot\\hat{\\vect k}/|B|$ at the nodes."),
    (("X, Y = magnetoionic_params", "nu_b = NU_B_PER_GAUSS"), "$X$, $Y$ and $\\nu_B$ at the nodes."),
    (("cut_o = X >= 1.0", "cut_x = X >= 1.0 - Y"), "Where each mode is cut off."),
    (("m = lambda a", "step_ok = "), "Mid-step values; a step counts if both nodes are inside the domain and it has non-zero length."),
    (("if use_ff:", "dtau_x = np.zeros_like(dl)"), "Free--free optical depth of each step for both modes, $\\kappa_\\sigma\\Delta\\ell$ at the mid-point."),
    (("st_o = dtau_o * te_m", "x_flag_step ="), "Accumulators: $\\sum\\tau T$ (to build the step's source function), layer optical depths, the strongest harmonic of each step and the validity flag."),
]),
("if use_gr:", "transfer_chunk(): harmonic layers", [
    ("cross = step_ok & ((s * f0 - freq)", "Layer $s$ lies in the step if $s\\nu_B-\\nu$ changes sign between its nodes."),
    (("a0, a1 = f0[ii, kk]", "ne_c, te_c, cos_c, X_c = lerp"), "Fractional position of the crossing and linear interpolation of $n_e$, $T_e$, $\\cos\\theta$, $X$ to it."),
    ("l_b = (freq / (s * NU_B_PER_GAUSS))", "$L_B=|B|/|dB/d\\ell|$ with $|B|$ = the layer value and the gradient from the two nodes."),
    (("t_x = layer_optical_depth", "t_o = layer_optical_depth"), "Optical depths of the layer in both modes (opacity.py)."),
    (("if s == 1:", "t_x = np.where(X_c > 0"), "At $s=1$ ($Y=1$) the x-mode is cut off for any $X>0$; it never reaches its fundamental layer."),
    (("np.add.at(gr_x", "np.add.at(st_o"), "Add to the step totals (several harmonics can fall in one step)."),
    (("tmax = np.maximum(t_x, t_o)", "x_flag_step[ii, kk] |="), "Remember the strongest harmonic of the step, and flag layers with $\\tau>0.1$ at $X>0.25$, where $n_\\sigma\\approx1$ is a poor approximation."),
]),
("dtx = dtau_x + gr_x", "transfer_chunk(): choice of the two polarisations", [
    "`h1`, `h2` carry $\\sum\\tau T$ of the layers (needed for the shares when an extra source is present).",
    (("e1 = e2 = None", "e1, e2 = e1s, e2s"), "Optional extra contribution (gyrosynchrotron in the combined image): optical depth and $\\tau T_\\text{eff}$ in $R$/$L$, mapped to the two polarisations (in strong coupling $x=R$ where $\\vect B\\cdot\\vect k>0$) and added to the step sums."),
    (("if coupling == \"weak\":", "c1, c2 = np.where(posn"), "Weak coupling: polarisation 1 = right-hand circular, which is the x-mode where $\\vect B\\cdot\\vect k>0$ and the o-mode elsewhere; polarisation 2 = left-hand. The cutoffs are chosen per node in the same way."),
    (("elif coupling == \"strong\":", "d1, d2, s1, s2"), "Strong coupling: polarisation 1 = x-mode, 2 = o-mode."),
]),
("out = {}", "transfer_chunk(): formal solution", [
    (("alive = np.cumprod", "g = np.where(ok, g, 0.0)"), "A polarisation stops at its first cutoff node: later steps are switched off."),
    ("tau_front = np.cumsum(d, axis=1) - d", "Optical depth between the observer and the start of each step."),
    ("S = np.where(d > 0, sw /", "Source function of the step: the $\\tau$-weighted mean of $\\nu^2k_BT/c^2$ over its free--free part and its layers."),
    (("w = np.exp(-tau_front)", "I = contrib.sum(axis=1)"), "$I=\\sum_k S_k(1-e^{-\\Delta\\tau_k})e^{-\\tau_{<k}}$, the exact solution for piecewise-constant $S$."),
    (("if t_body is not None", "I = I + np.where(body"), "Optional body (chromosphere) at temperature `t_body` behind rays that hit it."),
    (("contrib_tot += contrib", "contrib_gr += grp"), "Per-step contributions, total and from the layers; with an extra source the shares are those of the source function, $\\Delta\\tau_iS_i/\\sum\\Delta\\tau S$, for the layers and the extra part."),
    ("out[\"ff\" + tag]", "The free--free part of the polarisation (used by the refraction correction of the combined image)."),
]),
("if coupling == \"weak\":", "transfer_chunk(): Stokes V and diagnostics", [
    ("out[\"V\"] = out[\"I1\"] - out[\"I2\"]", "Weak coupling: $V=I_R-I_L$ (`vsign` = 1)."),
    (("has_b = bmag > 0", "out[\"V\"] = sgn"), "Strong coupling: $V=\\pm(I_x-I_o)$, sign of $\\vect B\\cdot\\vect k$ at the observer end of the ray."),
    (("tot = contrib_tot.sum", "out[\"gr_fraction\"]"), "Intensity-weighted emission radius and the fraction of $I$ coming from layers."),
    (("dom = np.full", "out[\"dominant_s\"] = dom"), "Dominant harmonic: the $s$ whose layers contribute most; 0 if free--free dominates, $-1$ if there is no emission."),
]),
])

entry("gyroresonance/image.py", "gyroresonance/image.py --- make_gr_image and GRImage",
r"""The driver: the same image geometry and observers as the refractive images, straight
rays, polarised transfer, and a `RadioImage` subclass with Stokes V and diagnostics.""",
[
('"""Gyroresonance + free-free images', "Module docstring and imports", []),
("class GRImage", "GRImage: attributes", [
    "Inherits intensity, brightness temperature, flux density, FITS/Tecplot output from `RadioImage`; adds Stokes V, the two polarisations, their optical depths and the diagnostics.",
    ("pol_names=(\"R\", \"L\")", "('R','L') for weak, ('x','o') for strong coupling."),
]),
("def _tb(self, a)", "GRImage: derived quantities", [
    ("def tb_v", "Stokes V as a brightness temperature."),
    ("def tb_pol", "Per-polarisation $T_p=I_pc^2/(k_B\\nu^2)$, equal to $T_e$ for a thick isothermal source."),
    ("def polarization", "$V/I$."),
    ("def flux_density_v", "Flux density of Stokes V."),
]),
("def summary(self)", "GRImage: summary and output", [
    ("def summary", "One line per frequency: peak $T_b$, flux in sfu and Jy, $V/I$ of the total flux, GR fraction and the dominant harmonics."),
    ("def save_npz", "All arrays, including V, the polarisations and the diagnostics."),
    ("def save_fits", "`'intensity'`, `'tb'` (as `RadioImage`), or `'v'`, `'tb_v'`, `'pol'`."),
]),
("def plot(self, quantity=\"tb\"", "GRImage.plot()", [
    "`'tb'`, `'intensity'` as before; `'v'` (symmetric colour scale), `'pol'` ($-1$ to 1), `'harmonic'`, `'radius'`, `'gr_fraction'`.",
]),
("def make_gr_image", "make_gr_image(): set-up", [
    (("if not getattr(grid, \"has_field\"", "raise ValueError(\"mechanisms must be"), "Check that the grid has a field and that `mechanisms` is valid."),
    (("xc, yc, dirs, tau_v", "pos, status0 = start_points(grid, observer"), "Image geometry and start points exactly as in `make_radio_image`, including the parallel (stellar) projection."),
    (("t_end, status = segment_ends", "hit_body = status == HIT_BODY"), "Length of each line and the step."),
]),
("nf = len(freqs)", "make_gr_image(): loop over frequencies and chunks", [
    (("for sm in iter_samples", "dom[k, i], cut[k, i], xf[k, i]"), "Sample a chunk, do the transfer, store the per-ray results."),
    (("st = np.broadcast_to", "st[cut & (st != MISSED)] = OVERDENSE"), "Rays stopped by a cutoff get the status 'over-dense'."),
    (("img = GRImage(", "pol_names="), "Assemble the image: $I=I_1+I_2$, $\\tau$ of polarisation 1 in `tau`, everything reshaped to (nfreq, ny, nx)."),
]),
])

entry("gyroresonance/__init__.py", "gyroresonance/__init__.py --- package interface",
r"""Registers the mechanism and exports its functions.""",
[('"""Thermal gyroresonance', "Mechanism registration", [
    "The docstring lists the three modules and what they take from `common/` and `bremsstrahlung/`.",
    ("make_image = make_gr_image", "Same name as in the other folders."),
    ("MECHANISM = register(Mechanism(", "`coefficients=None`: gyroresonance cannot be used inside the refractive tracer."),
])])


# ---------------------------------------------------------------------------
# common/fields.py and gyrosynchrotron/
# ---------------------------------------------------------------------------
entry("common/fields.py", "common/fields.py --- derived fields (current density)",
r"""Computes $\nabla\times\vect B$ and $|\vect J|$ on the grid, a proxy for where magnetic
energy is released, usable in fast-electron rules.""",
[
('"""Derived fields', "Module docstring and constants", [
    ("MU0 = ", "$\\mu_0$ [SI]."),
]),
("def _grad", "_grad()", [
    "Second-order `np.gradient` along one axis on a non-uniform axis; zero for a degenerate (single-cell) axis, e.g. $\\phi$ of an axisymmetric grid.",
]),
("def curl_b", "curl_b()", [
    (("if hasattr(grid, \"x\")", "return cx, cy, cz"), "Cartesian grid: $(\\nabla\\times\\vect B)_x=\\partial_yB_z-\\partial_zB_y$, etc."),
    (("R = grid.r[:, None, None]", "bp = -bx * sp + by * cp"), "Spherical grid: the components $(B_r,B_\\theta,B_\\phi)$ (converted if the field was given in Cartesian components)."),
    (("c_r = ", "c_p = "), "Spherical curl: $(\\nabla\\times\\vect B)_r=\\frac{1}{r\\sin\\theta}[\\partial_\\theta(\\sin\\theta B_\\phi)-\\partial_\\phi B_\\theta]$, $(\\nabla\\times\\vect B)_\\theta=\\frac1r[\\frac{1}{\\sin\\theta}\\partial_\\phi B_r-\\partial_r(rB_\\phi)]$, $(\\nabla\\times\\vect B)_\\phi=\\frac1r[\\partial_r(rB_\\theta)-\\partial_\\theta B_r]$."),
]),
("def current_density", "current_density()", [
    ("j = mag * 1e-4", "$|\\vect J|=|\\nabla\\times\\vect B|/\\mu_0$: gauss per coordinate unit $\\to$ T/m (factor $10^{-4}$/length unit), divided by $\\mu_0$, in A m$^{-2}$."),
    ("grid.add_scalar(\"j\", j)", "Stored as the grid scalar `'j'`, so rules can use `f['j']`."),
]),
])

entry("gyrosynchrotron/library.py", "gyrosynchrotron/library.py --- the fast GS codes as a library",
r"""Finds, builds and loads the public fast gyrosynchrotron codes (Kuznetsov \& Fleishman
2021). The codes are GPL-3 and are not part of radiosynth; this module only calls them
through `ctypes`.""",
[
('"""Access to the public', "Module docstring and imports", [
    "Describes the origin of the codes, where the library is searched for, and how `build_library` works.",
    (("REPO_URL", "DEFAULT_HOME"), "Repository and default installation folder (`~/.radiosynth/gyrosynchrotron`)."),
]),
("def _candidate_names", "_candidate_names()", [
    "Library file names produced by the repository's makefile or shipped prebuilt, for Windows (32/64-bit DLL), macOS (x86-64, arm64) and Linux.",
]),
("def find_library", "find_library()", [
    (("for base in (path,", "cands.append"), "Search order: explicit path, `RADIOSYNTH_GSCODES`, the default folder; each may be the file itself or a folder (or its `binaries/` subfolder)."),
    ("raise FileNotFoundError", "If nothing is found, explain how to build or point to the library."),
]),
("class GSLibrary", "GSLibrary: ctypes binding", [
    (("ip = ndpointer", "dp = ndpointer"), "Argument types: Fortran-ordered int32 and float64 arrays, as required by the codes."),
    (("rt = ctypes.c_int", "f.restype = rt"), "The exported `pyGET_MW` and `pyGET_MW_SLICE` return `double` on Linux/macOS and `int` on Windows (see the library's PythonInterface.cpp)."),
    (("def get_mw(self", "return RL"), "One line of sight: inputs converted to the right types and order, `RL` modified in place; a non-zero return code raises an error."),
    (("def get_mw_slice", "return RL_M"), "Many lines of sight (OpenMP inside the library)."),
]),
("def load_library", "load_library()", [
    "Loads each library file once and caches the binding.",
]),
("def build_library", "build_library()", [
    (("if not os.path.isdir(os.path.join(dest, \"source\"))", "subprocess.run([\"git\", \"-C\", dest, \"checkout\", ref]"), "Clone the repository (optionally a given commit or tag, for reproducibility)."),
    (("built = False", "built = r.returncode == 0"), "Compile with the repository's makefile."),
    (("if not built:", "print(\"compilation not possible"), "Otherwise fall back to the prebuilt binary of the repository, if allowed."),
]),
("def library_version", "library_version()", [
    "The git commit and date of the checkout, printed with every image so that results can be reproduced.",
]),
("if __name__ == \"__main__\"", "Command line", [
    "`python -m radiosynth.gyrosynchrotron.library --build [--ref COMMIT]` builds the codes; without options it prints the path found.",
]),
])

entry("gyrosynchrotron/electrons.py", "gyrosynchrotron/electrons.py --- the fast-electron prescription",
r"""`FastElectrons` holds the description of the fast electrons; the rule functions
build spatial distributions of their density.""",
[
('"""Description of the fast', "Module docstring", [
    "Lists the analytical energy and pitch-angle distributions of the codes and explains rules (functions of the sampled fields) and array-defined distributions.",
]),
("import math", "Index tables", [
    (("ENERGY_INDEX", "PITCH_INDEX"), "Names of the distributions and the indices the codes expect in `Parms[6]` and `Parms[14]` (AnalyticalDistributions.pdf)."),
    ("_trapz = ", "Trapezoidal integration (numpy 1 and 2)."),
]),
("class FastElectrons", "FastElectrons: constructor", [
    (("energy, pitch = energy.lower()", "raise ValueError(\"'kap' needs kappa\")"), "Check the distribution names and the parameters each one needs."),
    (("self.nb, self.energy", "self.array = None"), "Store everything; numbers and rules are both allowed."),
]),
("def from_array", "FastElectrons.from_array()", [
    (("norm = 2 * np.pi", "obj.array = "), "A tabulated $f(E,\\mu)$ normalised so that $2\\pi\\iint f\\,d\\mu\\,dE=1$; it is multiplied by the local $n_b$ when passed to the codes."),
]),
("def _eval", "Evaluation of numbers and rules", [
    ("def _eval", "A rule is called with the dict of sampled fields; a number is broadcast."),
    (("def evaluate", "return out"), "All parameters for the voxels of a chunk; $n_b$ below `nb_min` is set to zero, so the expensive gyrosynchrotron calculation is skipped there."),
]),
("def energy_density", "Energy check", [
    (("def moment(k)", "mean = moment(1) / moment(0)"), "For a power law, the mean energy $\\int E^{1-\\delta}dE/\\int E^{-\\delta}dE$ between the cutoffs (logarithm for the exponent $-1$)."),
    ("return nb * mean * 1.602176634e-6", "Energy density $n_b\\langle E\\rangle$ in erg cm$^{-3}$."),
    (("def check_energy", "return rmax"), "Ratio to $B^2/8\\pi$; a warning above `limit` (default 0.1)."),
]),
("def uniform(nb)", "Rules for n_b", [
    ("def uniform", "Constant wherever there is plasma."),
    ("def fraction_of_thermal", "$n_b=$ fraction $\\times n_e$ (with $n_e$ in cm$^{-3}$)."),
    ("def power_of", "$n_b=n_{b0}(q/q_\\text{ref})^p$ for any sampled quantity (e.g. `'bmag'`, `'j'`), zero below `floor`."),
    ("def in_sphere", "Gaussian blob truncated at `cutoff` radii, or a top-hat sphere."),
    ("def in_box", "Constant inside a box."),
    ("def product", "Product of rules and numbers."),
]),
])

entry("gyrosynchrotron/image.py", "gyrosynchrotron/image.py --- make_gs_image and GSImage",
r"""Builds the voxel arrays for the codes, calls them for chunks of lines of sight, and
assembles the image.""",
[
('"""Gyrosynchrotron (+ free-free) images', "Module docstring and imports", [
    "States the division of work between radiosynth and the codes and the conventions (viewing angle, voxel order, output rows, sfu $\\to$ intensity) that the tests check.",
    (("COUPLING_ROWS", "ABUNDANCE"), "Rows of the output array for each coupling and the abundance keys."),
]),
("class GSImage", "GSImage", [
    "A `GRImage` (Stokes I and V, polarisations, plotting, FITS); `gs_fraction` is the fraction of the intensity due to the fast electrons; harmonic and radius maps are not defined here.",
    ("def summary", "One line per frequency (sfu for the Sun, mJy for a star)."),
]),
("def _fields", "_fields(): voxel quantities", [
    "Mid-point values of position, $n_e$, $T_e$, $\\vect B$ and every grid scalar for all voxels of a chunk; this is the dict passed to the rules.",
]),
("def voxel_parms", "voxel_parms(): the parameters of the codes", [
    (("P = np.zeros((24, dl_cm.size))", "P[22] = abund"), "The 24 voxel parameters of the codes (CallingConventions.pdf): length, $T$, $n_e$, $B$, $\\theta$, mechanism flags (free--free on or off; electron--neutral collisions off), distribution indices and parameters, $n_p=n_e$, and the local key: analytical, array-defined, or free--free only where $n_b=0$. Shared with the combined image."),
]),
("def make_gs_image", "make_gs_image(): set-up", [
    (("if not getattr(grid, \"has_field\"", "raise ValueError(\"coupling must be"), "Checks."),
    ("lib = load_library(library)", "Load the codes (error message with build instructions if missing)."),
    (("xc, yc, dirs, tau_v", "hit_body = status == HIT_BODY"), "Geometry, start points, line lengths and voxel size, as in the gyroresonance driver."),
    (("s_pix = ", "to_int = SFU"), "Pixel area and the conversion $I=F\\,\\text{AU}^2/S$ of the codes' output."),
    (("arr = electrons.array", "log_key = 0 if"), "Energy and pitch-angle grids of an array-defined distribution; logarithmic or linear energy spacing."),
]),
("for sm in iter_samples", "make_gs_image(): one chunk of lines of sight", [
    (("fl = _fields", "nb = np.where(valid"), "Voxel quantities and fast-electron parameters; empty or zero-length voxels get no electrons."),
    (("if gyrosynchrotron and nb.max", "e_ratio = max"), "Energy check where there are fast electrons."),
    (("nobs = np.repeat", "theta = np.degrees"), "Viewing angle between $\\vect B$ and the direction to the observer."),
    ("P = voxel_parms(", "The 24 voxel parameters (`voxel_parms`, above)."),
    ("P = P.reshape(24, npix, nz)[:, :, ::-1]", "Reverse the voxel order: the codes integrate from the far end towards the observer."),
    (("if arr is not None:", "f_arr = np.zeros"), "Array-defined distribution: the normalised shape times $n_b$ in every voxel, in the (NE, Nmu, Nz, Npix) layout."),
    (("Lp = np.array", "Rp[3], Rp[4] = fc, fwh"), "Global integer and real parameters: dimensions, accuracy keys, source area, switch frequencies."),
    (("def run(Pm):", "return RL"), "One call of GET\\_MW\\_SLICE; frequencies given explicitly; a body behind the plasma enters as the intensity at the start of the line of sight."),
    (("RL = run(Parms_M)", "I_R[:, idx] ="), "Left and right intensities for the chosen coupling."),
    (("if diagnostics:", "I_ff[:, idx] ="), "Optional second call without fast electrons."),
]),
("shp = (nf, ny, nx)", "make_gs_image(): the image", [
    ("frac = np.where(I > 0, 1.0 - I_ff / I", "Fast-electron fraction $1-I_\\text{ff}/I$."),
    (("img = GSImage(", "pol_names="), "$I=I_R+I_L$, $V=I_R-I_L$; optical depths are not available (NaN)."),
]),
])

entry("gyrosynchrotron/__init__.py", "gyrosynchrotron/__init__.py --- package interface",
r"""Registers the mechanism and exports its functions.""",
[('"""Gyrosynchrotron emission', "Mechanism registration", [
    "The docstring lists the three modules, the licence of the codes and a short example.",
    ("make_image = make_gs_image", "Same name as in the other folders."),
    ("MECHANISM = register(Mechanism(", "Registry entry; `coefficients=None` (not usable in the refractive tracer)."),
])])


# ---------------------------------------------------------------------------
# combined/
# ---------------------------------------------------------------------------
entry("combined/coefficients.py", "combined/coefficients.py --- gyrosynchrotron coefficients per voxel",
r"""Extracts the gyrosynchrotron optical depth and emission of single voxels from the fast
GS codes, which by themselves only return intensities at the end of a line of sight.""",
[
('"""Per-voxel gyrosynchrotron', "Module docstring and constants", [
    "Explains the two one-voxel calls ($I_\\text{in}=0$ and a known $I_\\text{in}$) and why the codes' free--free is switched off.",
    ("S_CM2 = ", "Nominal source area of the calls; it cancels in the intensity."),
]),
("def gs_voxel_coefficients", "gs_voxel_coefficients()", [
    (("Lp = np.array", "fa = np.asfortranarray"), "One GET\\_MW\\_SLICE call treats every voxel as its own one-voxel line of sight; array-defined distributions are passed per voxel."),
    (("def run(I0):", "return RL[1].copy(), RL[2].copy()"), "One call, with or without incoming intensity; rows 1/2 (left/right, weak coupling: no coupling inside one voxel)."),
    (("E_L, E_R = run(None)", "O_L, O_R = run(I0)"), "Emission $E$ alone, then with $I_\\text{in}=10^6E$ (large, so that $e^{-\\tau}$ is resolved to round-off)."),
    (("att = np.clip", "out[\"E_\" + pol]"), "$e^{-\\tau}=(I_\\text{out}-E)/I_\\text{in}$, $\\tau$ capped at 700; $E$ converted from sfu to intensity."),
]),
("def source_weights", "source_weights()", [
    "$\\tau T_\\text{eff}=\\tau E/[(1-e^{-\\tau})\\,\\nu^2k_B/c^2]$, the quantity added to the step sums of the transfer (limit $E/(\\nu^2k_B/c^2)$ for $\\tau\\to0$).",
]),
])

entry("combined/image.py", "combined/image.py --- make_combined_image and CombinedImage",
r"""The combined driver: straight rays, radiosynth's free--free and layers, the
gyrosynchrotron coefficients of the voxels with fast electrons, one polarised transfer,
and the optional refraction correction of the free--free part.""",
[
('"""Combined free-free', "Module docstring and imports", [
    "Describes how the three processes are combined and the refraction correction $I_p\\to I_p+I_{\\text{ff},p}(R-1)$ of the free--free part.",
    ("MECH_NAMES", "Codes of the dominant-mechanism map."),
]),
("class CombinedImage", "CombinedImage", [
    (("def __init__", "-1).astype(np.int16)"), "Free--free share $=1-$ (layers + gyrosynchrotron); the dominant process is the largest share."),
    ("def summary", "Flux, $V/I$ and the flux-weighted shares of the three processes per frequency."),
    (("def plot", "return super().plot"), "Extra plots: `mechanism`, `ff_fraction`, `gs_fraction`."),
    (("def _plot_extra", "return fig"), "Categorical map of the dominant process, or a fraction map."),
    ("def save_npz", "Adds the shares and the dominant-process map to the saved file."),
]),
("def make_combined_image", "make_combined_image(): set-up", [
    (("include = tuple", "mech = \"+\".join"), "Which processes; the free--free and layer part is passed to the gyroresonance transfer as its `mechanisms` string."),
    (("if use_gs:", "log_key = 0 if"), "Fast-electron machinery from the gyrosynchrotron folder (library, distribution indices, array-defined distribution)."),
    (("xc, yc, dirs, tau_v", "hit_body = status == HIT_BODY"), "Geometry, straight lines and step, exactly as in `make_gr_image` (so that without fast electrons the result is identical)."),
]),
("for sm in iter_samples", "make_combined_image(): one chunk", [
    (("fl = _fields", "sel = np.flatnonzero"), "Voxels with fast electrons."),
    (("P = voxel_parms(", "False, eidx, pidx, -1, arr is not None)"), "Their parameters for the codes, with free--free switched off."),
    (("if arr is not None:", "q_optimization=q_optimization)"), "Per-voxel coefficients for all frequencies in one go."),
    (("sw = source_weights", "extra_all[key] = a.reshape"), "Scatter them onto the (frequency, ray, step) grid."),
    (("for k, f in enumerate(freqs):", "dom[k, idx], cut[k, idx]"), "One polarised transfer per frequency with the extra contribution."),
]),
("ff_factor = None", "make_combined_image(): refraction correction of the free-free part", [
    (("i_ref = make_radio_image", "i_los = make_radio_image"), "Free--free images from the refractive tracer, with and without refraction (same Gaunt factor, full transfer)."),
    ("ff_factor = np.where(good", "$R$ per pixel where the straight free--free image is not negligible."),
    (("d1 = np.where(good", "res[\"V\"] = "), "$I_p\\to I_p+I_{\\text{ff},p}(R-1)$; where $R$ is undefined, half of the difference is added to each polarisation; $V$ follows."),
    (("for key in (\"gr_fraction\", \"gs_fraction\")", "res[key] = np.where"), "The shares are renormalised to the new intensity."),
]),
("shp = (nf, ny, nx)", "make_combined_image(): the image", [
    ("img = CombinedImage(", "Assemble the image; `ff_refraction_factor` holds $R$."),
]),
])

entry("combined/__init__.py", "combined/__init__.py --- package interface",
r"""Registers the combined mechanism.""",
[('"""Combined image', "Mechanism registration", [
    "The docstring lists the modules and what they use from the other folders.",
    ("make_image = make_combined_image", "Same name as in the other folders."),
    ("MECHANISM = register(Mechanism(", "Registry entry."),
])])


# ---------------------------------------------------------------------------
# 1D (spherically symmetric) models
# ---------------------------------------------------------------------------
entry("common/symmetric.py", "common/symmetric.py --- 1D (spherically symmetric) models",
r"""Turns a radial profile into a grid, and computes images, radial brightness profiles and
fluxes of a symmetric model from one row of rays.""",
[
('"""Spherically symmetric', "Module docstring and imports", [
    "Explains the axisymmetric grid with one colatitude cell, the one-row method and why only the mechanisms of the refractive tracer are offered.",
]),
("def grid_from_profile", "grid_from_profile()", [
    (("r = np.asarray(r, float)", "raise ValueError(\"r must be 1D"), "Check the radii."),
    (("g = SphericalGrid(", "theta_edges=np.array([0.0, np.pi])"), "One colatitude cell with edges $0$ and $\\pi$ and no longitude: the fields depend on $r$ only, and every direction is inside the domain."),
    ("g.symmetric = True", "Mark the grid as symmetric."),
]),
("def read_profile", "read_profile(): text tables", [
    ("d = np.loadtxt", "One radius per line; columns chosen by the caller."),
    (("if density == \"ne\":", "raise ValueError(\"density must be"), "Electron density in cm$^{-3}$ or m$^{-3}$, or mass density converted with $n_e=\\rho/(\\mu_em_p)$."),
    (("order = np.argsort(r)", "return grid_from_profile"), "Sort by radius and build the grid."),
]),
("def is_symmetric", "is_symmetric()", [
    "A grid made by `grid_from_profile`, or any spherical grid with one $\\theta$ and one $\\phi$ cell (e.g. reloaded from npz or read from PLUTO 1D output).",
]),
("def _rotate", "_rotate()", [
    "Linear interpolation of the profile at the distance $\\sqrt{X^2+Y^2}$ of every pixel; zero beyond the last ray.",
]),
("def make_symmetric_image", "make_symmetric_image(): one row of rays", [
    (("if p_max is None:", "p_max = float(getattr"), "Default $p_\\text{max}$: half the image diagonal, or the outer radius of the grid."),
    (("dp = p_max / n_p", "row = dict("), "A $(2n_p+1)\\times1$ image whose pixel centres are $p=k\\,\\Delta p$, $k=-n_p\\ldots n_p$: one ray exactly at the centre."),
    ("kw.setdefault(\"r_integration\"", "Integrate along the whole line of sight through the model."),
    (("if callable(mech) or mech in", "magnetic field\")"), "`make_radio_image` for bremsstrahlung, simplistic or a user emission function (refraction on or off as requested); other mechanisms are refused."),
]),
("nf = len(im.freqs)", "make_symmetric_image(): rotation and flux", [
    (("half = slice(n_p, 2 * n_p + 1)", "p[0] = 0.0"), "The rays with $p\\ge0$."),
    (("for name, val in list(vars(im).items()):", "profile[name] = v"), "Every per-pixel array of the image is rotated: floating-point quantities by linear interpolation in $p$, status codes from the nearest ray."),
    ("im.meta = dict(", "Record the 2D image layout and the symmetric settings."),
    (("trap = getattr", "flux = 2.0 * math.pi"), "$S=2\\pi D^{-2}\\int_0^{p_\\text{max}}I(p)\\,p\\,dp$ (trapezoid rule) over the whole circle."),
    ("profile.update(", "Profile: $p$, $I(p)$, $T_b(p)$, flux."),
]),
("def plot_profile", "plot_profile()", [
    "$T_b(p)$ for every frequency on a logarithmic scale, with the limb marked.",
]),
])

# ---------------------------------------------------------------------------
# common/raytrace.py: the tracer, organised in subroutines
# ---------------------------------------------------------------------------
entry("common/raytrace.py", "common/raytrace.py --- the ray tracer",
r"""This is the core of the package. It is written following the refractive ray tracing of
the BATS-R-US radio-wave imaging module (`ray_path` in `ModRadioWaveImage.f90`;
Benkevitch et al. 2010; Moschou et al. 2018), but it advances \emph{all} rays together:
every quantity is an array over the currently active rays, and the decisions taken per ray
become boolean masks.

The file is organised in small subroutines, one per task, and the main function
`trace_rays` calls them in order in each iteration. Each subroutine acts on the rays of the
current iteration only. Index conventions used throughout: `act` holds the global indices of
the rays still active; arrays such as `p`, `ne`, `eps` are \emph{local} to `act`; `adv`
holds local indices of the rays that advance in this iteration, and `ia = act[adv]` their
global indices. Subroutines that change the state of rays (`ds`, `slope`, `steep`,
`intensity`, ...) receive the global arrays and the global indices.""",
[
('"""Refractive', "Module docstring and imports", [
    "Lists the ray equations and the subroutines called in one step, in order.",
]),
("# Termination codes", "Ray status codes", [
    ("ACTIVE, ESCAPED", "Integer codes stored per ray: 0 active, 1 escaped, 2 hit body, 3 over-dense, 4 $\\varepsilon\\le0$, 5 optically thick, 6 step limit, 7 missed the grid."),
    ("STATUS_NAMES", "Names used in printed summaries."),
]),
("def _emission_function", "_emission_function(): the emission model as a function", [
    (("if hasattr(emission, \"kind\")", "return emit"), "Built-in emission models receive the length unit (needed by the simplistic mechanism, which is defined per normalised length)."),
    ("return emission  ", "User functions are called as `f(ne, te, freq)` and must return $(\\kappa, B)$."),
]),
("def _add_emission", "_add_emission(): radiative transfer over a segment", [
    ("intensity[idx] += kappa * B * dl", "Optically thin: $I\\mathrel{+}=\\kappa B\\Delta\\ell$, as in BATS-R-US."),
    (("dtau = kappa * dl", "tau[idx] += dtau"), "Full transfer: add $B e^{-\\tau}(1-e^{-\\Delta\\tau})$, the exact solution for constant $\\kappa$, $B$ on the segment, then update $\\tau$. `-expm1(-x)` is $1-e^{-x}$, accurate for small $x$."),
]),
("def _predictor", "_predictor(): half step and plasma at the mid point", [
    ("half = 0.5 * d", "$\\Delta s/2$."),
    ("pm = p + s * half", "Predictor: $\\vect r_{1/2}=\\vect r_0+\\vect v_0\\Delta s/2$."),
    ("ne, gn, te, cell, inside = grid.sample", "Density, its gradient, temperature, local cell size and an inside-the-domain flag at $\\vect r_{1/2}$."),
]),
("def _terminations", "_terminations(): rays that end at the mid point", [
    ("r2 = np.einsum", "$|\\vect r_{1/2}|^2$ (`einsum` gives row-wise dot products)."),
    ("term = np.zeros", "Termination code (0 = continue)."),
    ("term[r2 < rbody2] = HIT_BODY", "Mid point inside the body: the ray reached the photosphere. Tested first, because for a spherical grid the body is also outside the domain."),
    ("term[(term == 0) & (~inside", "Left the domain or the integration sphere: escaped."),
    ("term[(term == 0) & (ne >= ncr)]", "Over-dense plasma at the mid point; in the default mode such rays are recovered by `_snell_fallback`."),
]),
("def _integrate_to_body", "_integrate_to_body(): the last piece before the photosphere", [
    "When the mid point is already inside the body, the piece of the step between $\\vect r_0$ and the surface would otherwise be lost. It is integrated as a straight segment.",
    (("bb = np.einsum", "tt = np.clip"), "Distance to the sphere $r_\\text{body}$ along $\\vect v$: $t=-b-\\sqrt{b^2-c}$ with $b=\\vect r_0\\cdot\\vect v$ and $c=r_0^2-r_\\text{body}^2$, limited to $[0,\\Delta s]$."),
    (("ne_h, _, te_h", "k_h, B_h = emit"), "Emission at the mid point of that piece; the density is capped at $n_\\text{cr}$."),
    ("_add_emission(intensity", "Add it, in metres, in the thin or full mode."),
]),
("def _dielectric", "_dielectric(): permittivity and its gradient", [
    ("eps = 1.0 - ne / ncr", "$\\varepsilon_{1/2}=1-n_e/n_\\text{cr}$ at the mid point."),
    ("ge = -gn / ncr", "$\\nabla\\varepsilon=-\\nabla n_e/n_\\text{cr}$ per coordinate unit."),
    ("ge2 = np.einsum", "$|\\nabla\\varepsilon|^2$."),
    ("gs = np.einsum", "$\\nabla\\varepsilon\\cdot\\vect v_0$: negative when the ray moves towards denser plasma."),
    ("ehb = eps - gs * half", "Back-extrapolated permittivity at the step start, $\\varepsilon_0=\\varepsilon_{1/2}-\\nabla\\varepsilon\\cdot\\vect v_0\\,\\Delta s/2$ (Benkevitch et al. 2010, Eq. 34)."),
]),
("def _snell_fallback", "_snell_fallback(): rays that crossed the critical surface (default mode)", [
    "Implements the fallback described by Benkevitch et al. (2010, Sec. 5.2) and Moschou et al. (2018, Sec. 2.2): return to the previous position, then reflect.",
    ("bad = (term == OVERDENSE)", "Rays that are over-dense at the mid point or have $\\varepsilon_0\\le0$."),
    (("back = bad & (d >= 4.0 * min_ds)", "ds[act[back]] = 0.25"), "First choice: retry from the same start point with a quarter of the step. The position is not changed, so the next iteration probes a point closer to $\\vect r_0$."),
    (("snell = bad & ~back", "steep[act[snell]] = False"), "If the step is already tiny, reflect the direction specularly about the gradient, $\\vect v\\leftarrow\\vect v-2(\\vect v\\cdot\\hat g)\\hat g$ with $\\hat g=\\nabla\\varepsilon/|\\nabla\\varepsilon|$, when the ray moves into the gradient. The step is reset to at least `step_min` and the steep flag cleared."),
    (("term[back | snell] = 0", "return back | snell"), "These rays stay active but do not advance in this iteration; the mask is returned to the caller."),
]),
("def _check_step", "_check_step(): precision control", [
    "Applied to rays that are not marked steep, and only with refraction on.",
    ("chk = ok & ~steep[act]", "Rays to check."),
    ("curv = (0.5 * half / ehb)**2", "$(\\Delta s/4\\varepsilon_0)^2(|\\nabla\\varepsilon|^2-(\\nabla\\varepsilon\\cdot\\vect v)^2)$, the squared turning angle over half a step; the bracket is the squared transverse gradient."),
    ("r_a = chk & (curv >= tol2)", "(a) Too much bending for the tolerance (about $1/(2\\,\\text{Tol})=50$ steps per radian)."),
    ("newd[r_a] = 0.99", "Shrink the step by the factor needed to meet the tolerance, with a 1\\% margin."),
    (("if not strict_batsrus:", "r_a &= d > min_ds"), "Default mode: never go below `min_ds`, and stop retrying once there."),
    ("r_b = chk & ~r_a & (3.0 * gs * half <= -ehb)", "(b) Correctness criterion (Benkevitch et al. 2010, Eq. 33): a linear prediction puts $\\varepsilon=0$ within about 1.5 steps."),
    ("d2c = ehb[r_b]", "Distance to the critical surface $d_\\text{cr}=\\varepsilon_0/|\\nabla\\varepsilon|$."),
    (("steep[idx] = True", "dist2cr[idx] = d2c"), "Mark the ray steep and remember $d_\\text{cr}$."),
    ("newd[r_b] = np.maximum", "New step $\\max(\\tfrac12\\text{Tol}\\,d_\\text{cr}, \\text{step\\_min})$."),
    (("retry = r_a | r_b", "return retry"), "These rays redo the step with the smaller $\\Delta s$; their position stays at $\\vect r_0$."),
]),
("def _parabolic_step", "_parabolic_step(): reflection at the critical surface", [
    ("lcos = -gs / ge2", "$L\\cos\\alpha=-\\nabla\\varepsilon\\cdot\\vect v/|\\nabla\\varepsilon|^2$, with $L=1/|\\nabla\\varepsilon|$ and $\\alpha$ the incidence angle."),
    ("proj = -lcos[:, None] * g", "Component of $\\vect v$ along $\\nabla\\varepsilon$: $\\vect v_\\parallel=-L\\cos\\alpha\\,\\nabla\\varepsilon$."),
    ("stepy = s - proj", "Transverse component $\\vect v-\\vect v_\\parallel$, of length $\\sin\\alpha$."),
    ("snew = s - 2.0 * proj", "Reflected direction: the parallel component is reversed."),
    ("stepy = 4.0 * stepy", "Scale to the chord of the parabola, $|\\Delta\\vect r_\\perp|=4\\sin\\alpha\\cos\\alpha\\,\\varepsilon_0 L$."),
    ("pnew = p_start + stepy", "Jump from $\\vect r_0$ to the symmetric point on the other branch."),
    ("stepx = (ehb * lcos**2)", "Depth of the parabola vertex along $\\nabla\\varepsilon$, $\\varepsilon_0(L\\cos\\alpha)^2|\\nabla\\varepsilon|$."),
    (("seg = np.sqrt", "np.einsum(\"ij,ij->i\", stepy, stepy))"), "Path length assigned to the arc, $\\sqrt{|2\\Delta\\vect r_\\parallel|^2+|\\Delta\\vect r_\\perp|^2}$."),
]),
("def _boris_step", "_boris_step(): Boris (CYLRAD) rotation", [
    ("coef = 0.5 * half / eps", "$\\Delta s/(4\\varepsilon_{1/2})$."),
    ("rel = coef[:, None] * g", "$\\vect a=\\nabla\\varepsilon\\,\\Delta s/(4\\varepsilon)=(\\nabla\\mu/\\mu)\\,\\Delta s/2$."),
    (("om = np.cross(rel, s)", "s1 = s + np.cross(s, om)"), "First estimate: $\\vect\\Omega_0=\\vect a\\times\\vect v_0$, $\\vect v'=\\vect v_0+\\vect v_0\\times\\vect\\Omega_0$."),
    (("om = np.cross(rel, s1)", "s1 = s + np.cross(s, om)"), "Rotation vector at the half step $\\vect\\Omega_{1/2}=\\vect a\\times\\vect v'$, and $\\vect v''=\\vect v_0+\\vect v_0\\times\\vect\\Omega_{1/2}$."),
    (("c2 = 2.0 /", "snew = s + c2"), "Exact-length rotation $\\vect v_1=\\vect v_0+\\frac{2}{1+\\Omega^2}\\vect v''\\times\\vect\\Omega_{1/2}$, so $|\\vect v_1|=1$."),
    ("pnew = pm + snew", "Corrector: $\\vect r_1=\\vect r_{1/2}+\\vect v_1\\Delta s/2$."),
]),
("def _advance", "_advance(): move every ray", [
    ("return s, pm + s * half", "Without refraction: direction unchanged, $\\vect r_1=\\vect r_{1/2}+\\vect v\\Delta s/2$, segment length $\\Delta s$."),
    ("refl = 3.0 * gs * half <= -ehb", "The parabolic switch is used where the linear prediction says the ray would reach $\\varepsilon=0$ (Eq. 33); the other rays take a Boris step."),
    (("if strict_batsrus:", "refl |= (d < step_min)"), "In strict mode the switch is also used when $\\Delta s<$ `step_min` and $\\nabla\\varepsilon\\cdot\\vect v<0$ (Section \\ref{sec:strict})."),
    (("if refl.any():", "p_start[refl], s[refl]"), "Parabolic switch for the selected rays."),
    (("bo = ~refl", "snew[bo], pnew[bo] = _boris_step"), "Boris step for the others."),
]),
("def _clip_at_body", "_clip_at_body(): a step that ends inside the photosphere", [
    ("into_body = r2e < rbody2", "Steps whose end point lies inside $r_\\text{body}$."),
    (("ps, dd_ =", "frac[into_body]"), "Solve $|\\vect r_0+t(\\vect r_1-\\vect r_0)|=r_\\text{body}$ for the first root $t\\in[0,1]$ and keep only that fraction of the segment; this matters at the disk centre at high frequency."),
]),
("def _new_step_size", "_new_step_size(): step regrowth", [
    ("cond = ~steep[ia]", "Regrow the step for gentle rays, and for steep rays that are now farther from the critical surface than when they were marked ($\\varepsilon_0>d_\\text{cr}|\\nabla\\varepsilon|$)."),
    ("cond[:] = True", "Without refraction, always regrow."),
    ("steep[ic] = False", "Those rays are gentle again."),
    ("ds[ic] = np.maximum", "Non-linear update $\\Delta s'=\\max(2-\\Delta s/\\Delta s_\\text{new},1)\\min(\\Delta s_\\text{new},\\Delta s)$: small steps roughly double each time and saturate smoothly at $\\Delta s_\\text{new}$."),
]),
("def trace_rays(", "trace_rays(): signature and docstring", [
    ("def trace_rays(", "Grid, starting points, directions, frequency and emission model are required; the rest are keyword options. The defaults `tolerance=1e-2` and `step_min=1e-2` are those of BATS-R-US."),
    ("strict_batsrus", "Switch between the original BATS-R-US step logic and the corrected default; explained in the docstring and in Section \\ref{sec:strict}."),
]),
("if transfer not in", "trace_rays(): set-up", [
    (("pos = np.array", "slope /= np.linalg.norm"), "Copies of positions $\\vect r$ and unit directions $\\vect v$; the directions are re-normalised."),
    ("ncr = float(critical_density", "Critical density for this frequency."),
    ("tol2 = tolerance**2", "Squared tolerance."),
    ("rint2 =", "Squared integration-sphere radius plus 0.01; infinite if none."),
    ("rbody2", "Squared inner-boundary radius."),
    ("L = float(getattr(grid", "Metres per coordinate unit: $R_\\odot$ for the Sun, $R_\\ast$ for a star."),
    ("emit = _emission_function", "The emission model as a function of $(n_e, T_e, \\nu)$."),
    ("status = np.zeros", "Initial status (rays that miss the grid arrive here already flagged)."),
    (("intensity = np.zeros", "ds = np.zeros"), "Per-ray state: intensity, optical depth, step count, distance to the critical surface, steep flag, step size."),
    ("act = np.flatnonzero", "Indices of the rays to trace."),
    (("if ds_init is None", "ds[:] = ds_init"), "Initial step: the local cell size at the entry point (times `step_factor`), or a user-given value."),
    (("paths = None", "slopes = {int(i)"), "Optionally record the position and direction of selected rays after every accepted step (used by `trace_paths` and the tests)."),
]),
("# ---- one iteration per loop", "trace_rays(): the main loop", [
    "One pass of the loop is one attempted step of every active ray. Each line calls one of the subroutines above.",
    ("while act.size:", "Iterate until every ray has ended."),
    (("p, s, d = pos[act]", "dsnew = step_factor * cell"), "Local copies of start point, direction and step; predictor; desired new step from the local cell size."),
    (("term = _terminations", "act[hb], intensity, tau, transfer)"), "Rays that end at the mid point; for those that reached the body, the remaining piece is integrated."),
    (("eps, ge, ge2, gs, ehb = _dielectric", "term[(term == 0) & (ehb <= 0.0)]"), "Permittivity and its gradient; $\\varepsilon_0\\le0$ means the ray is already beyond the critical surface."),
    (("if not strict_batsrus and not no_refraction:", "ok = (term == 0) & ~ok_bad"), "Default mode: recover the rays that crossed the critical surface; `ok` are the rays that continue normally."),
    (("retry = np.zeros", "tolerance, tol2, step_min"), "Step control (only with refraction)."),
    (("adv = np.flatnonzero", "ia = act[adv]"), "The rays that passed all checks advance now."),
    (("kap, B = emit", "kap = kap / np.sqrt"), "$\\kappa$ and $B$ at the mid point, optionally $\\kappa/\\mu$ with $\\mu=\\sqrt\\varepsilon$."),
    (("snew, pnew, seg = _advance", "strict_batsrus, step_min)"), "Move the rays (straight, parabolic switch or Boris)."),
    (("frac, into_body = _clip_at_body", "_add_emission(intensity, tau, ia"), "Clip at the photosphere and add the emission of the segment, in metres."),
    (("pos[ia] = pnew", "slope[ia] = snew"), "Store the new state."),
    (("r2n = np.einsum", "status[ia[(~gone) & (tau[ia] > tau_max)]] = OPAQUE"), "End-of-step tests: escaped, ended on the photosphere, or (full transfer) $\\tau>\\tau_\\text{max}$, after which nothing more can be seen."),
    (("_new_step_size(ia", "no_refraction)"), "Regrow the step."),
    (("if paths is not None:", "slopes[int(i)].append"), "Record the paths of the selected rays."),
    (("tmask = (term > 0)", "status[act[tmask]] = term[tmask]"), "Store the termination codes decided at the mid point."),
    ("nstep[act] += 1", "Count the iteration (including retries)."),
    (("over =", "status[act[over]] = MAX_STEPS"), "Safety stop for runaway rays."),
    ("act = act[status[act] == ACTIVE]", "Keep only the rays that are still active."),
]),
('out = {"intensity"', "trace_rays(): return value", [
    ("out = {", "Intensity [W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$], optical depth, status and step count per ray."),
    (("if paths is not None", "out[\"slopes\"]"), "Recorded trajectories as (M,3) arrays."),
]),
])

# ---------------------------------------------------------------------------
ORDER = [
    "common/constants.py", "common/magnetoionic.py", "common/grid.py", "common/readers.py",
    "common/pluto.py", "common/fields.py", "common/mechanism.py", "common/emission.py", "common/raytrace.py",
    "common/straight.py", "common/image.py", "common/symmetric.py", "common/figures.py", "common/stellar.py",
    "common/fitsio.py", "common/cli.py",
    "simplistic/emission.py", "simplistic/__init__.py",
    "bremsstrahlung/emission.py", "bremsstrahlung/magnetized.py", "bremsstrahlung/wind.py",
    "bremsstrahlung/__init__.py",
    "gyroresonance/opacity.py", "gyroresonance/transfer.py", "gyroresonance/image.py",
    "gyroresonance/__init__.py",
    "gyrosynchrotron/library.py", "gyrosynchrotron/electrons.py", "gyrosynchrotron/image.py",
    "gyrosynchrotron/__init__.py",
    "combined/coefficients.py", "combined/image.py", "combined/__init__.py",
    "__init__.py", "__main__.py",
]
assert set(ORDER) == set(M), set(M) ^ set(ORDER)
MODULES = [tuple(M[k]) for k in ORDER]
