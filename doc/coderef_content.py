# -*- coding: utf-8 -*-
# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Commentary for the line-by-line code reference.

MODULES = [(file, chapter title, intro, [(block start marker, heading, items)])]
items: paragraph strings, or (marker | (marker1, marker2), text) for a line entry.
Text uses LaTeX; `...` is typeset as code.
"""

MODULES = []

# ============================================================================
MODULES.append(("constants.py", "constants.py --- physical constants",
r"""All numbers in SI units. The solar radius is the BATS-R-US value (6.96e8 m), so that
lengths in $R_\odot$ convert exactly as in the Fortran code.""",
[
('"""Physical constants', "Module header and imports", [
    ('from scipy import constants', "`scipy.constants` provides CODATA values of $k_B$, $m_e$, $m_p$, $e$, $c$, $\\varepsilon_0$."),
]),
("PI = np.pi", "Constant definitions", [
    ("K_B = _c.k", "Boltzmann constant [J/K]; used in $B_\\nu$ and in $T_e=P/(n k_B)$."),
    (("M_E = _c.m_e", "C_LIGHT"), "Electron and proton masses, elementary charge, speed of light."),
    ("E2_JM", "$e^2/(4\\pi\\varepsilon_0)$ in J m. This is `cElectronChargeSquaredJm` of BATS-R-US and appears in $n_\\text{cr}$ and in $\\kappa_\\nu$."),
    ("R_SUN = 6.96e8", "Solar radius used to convert path lengths in $R_\\odot$ to metres."),
    (("AU = ", "AU_RSUN"), "1 AU in metres and in solar radii (about 215); a natural observer distance."),
    (("JANSKY", "SFU"), "Flux units: 1 Jy $=10^{-26}$ and 1 sfu $=10^{-22}$ W m$^{-2}$ Hz$^{-1}$."),
]),
("def critical_density", "critical_density()", [
    ("return PI * M_E", "Eq. (\\ref{eq:ncr}): $n_\\text{cr}=\\pi m_e f^2/(e^2/4\\pi\\varepsilon_0)$, identical to `NElectronSiCr = cPi*cElectronMass*RadioFrequency**2/cElectronChargeSquaredJm`. It accepts scalars or arrays."),
]),
("def plasma_frequency", "plasma_frequency()", [
    ("return np.sqrt", "The inverse relation, $f_p=\\sqrt{n_e e^2/(4\\pi\\varepsilon_0\\,\\pi m_e)}\\approx 8.98\\sqrt{n_e[\\text{m}^{-3}]}$ Hz."),
]),
]))

# ============================================================================
MODULES.append(("emission.py", "emission.py --- emission and absorption",
r"""Defines `EmissionModel`, a callable object that returns the absorption coefficient
$\kappa$ [m$^{-1}$] and source function $B$ for arrays of $n_e$ and $T_e$ at one frequency.
It is the Python counterpart of `user_material_properties` in `ModUserAwsom.f90`.
You can replace it with any function with the same signature.""",
[
('"""Radio emission', "Module docstring", [
    "Summarises the two mechanisms implemented in BATS-R-US. The formulas are written out in the User Manual, Part II.",
]),
("import numpy as np", "Imports and the bremsstrahlung prefactor", [
    ("_BREMS_PREF", "Pre-computes the constant part of $\\kappa$: $\\frac43\\sqrt{2\\pi/3}\\,(e^2/4\\pi\\varepsilon_0)^3/c$. This is the Fortran expression `4/3*sqrt(2*cPi/3)*(cElectronChargeSquaredJm/ ...)**3 /cLightSpeed` with the temperature factor taken out."),
]),
("class EmissionModel", "Class definition and constructor", [
    ("def __init__", "`kind` selects the mechanism; `gaunt` is a number (default 10, the value hard-coded in `ModUserAwsom.f90`) or the string 'dulk'."),
    (("kind = kind.lower()", "raise ValueError"), "Case-insensitive name; an unknown mechanism raises an error, as `CON_stop` does in Fortran."),
    ("def __repr__", "Readable description, printed by `make_radio_image`."),
]),
("def __call__", "Evaluation: __call__(ne, te, freq)", [
    ("ne = np.asarray", "Accept lists or arrays; all arithmetic in double precision."),
    (("if self.kind == \"simplistic\"", "return kappa, np.ones_like"), "Simplistic mechanism: $r=n_e/n_\\text{cr}$ and $\\kappa=r^2(0.5-r)^2$ per normalised length. Dividing by `length_unit` ($R_\\odot$, or $R_\\ast$ for a star) turns it into a per-metre coefficient, so that $\\kappa\\,\\Delta\\ell$[m] equals the Fortran product of $\\kappa$ and the normalised step. $B=1$, a proxy as in Fortran."),
    ("te = np.maximum", "Protects against zero or negative temperatures from bad cells. $T_e^{-3/2}$ would otherwise diverge."),
    ("B = 2.0 * freq", "Rayleigh--Jeans source function $B_\\nu=2\\nu^2k_BT_e/c^2$ (`PlanckOut_W`)."),
    (("if isinstance(self.gaunt, str)", "kappa = 9.78e-3"), "Dulk (1985) form: density converted to cm$^{-3}$, Coulomb logarithm $G$ chosen by temperature regime and floored at 1, $\\kappa$ in cm$^{-1}$ times 100 to give m$^{-1}$."),
    (("kappa = (_BREMS_PREF", "* (ne / freq)"), "BATS-R-US form: prefactor $\\times(k_BT_em_e)^{-3/2}\\times(n_e/\\nu)^2\\times\\langle g\\rangle$, the same expression as `OpacityEmissionOut_W`, but in m$^{-1}$ (the Fortran multiplies by $R_\\odot$ to get a per-normalised-length value)."),
    ("return kappa, B", "Both arrays have the shape of the input."),
]),
]))

# ============================================================================
MODULES.append(("grid.py", "grid.py --- model grids and interpolation",
r"""The ray tracer asks a grid object for five things at arbitrary points: $n_e$,
$\nabla n_e$, $T_e$, the local cell size, and whether the point is inside the domain.
`MHDGrid` (Cartesian, rectilinear) and `SphericalGrid` ($r,\theta,\phi$) provide the same
interface: `sample(p)`, `inside(p)`, `domain_intersection(o, d)`, `r_body`, `to_npz()`.
This replaces BATS-R-US's `get_local_density`, `interpolate_grid_amr_gc`,
`calc_gradient_ghost` and `CellSize_DB`.""",
[
('"""Cartesian', "Module docstring and import", [
    "Explains the correspondence with the BATS-R-US interpolation routines. The density gradient is pre-computed on the grid and interpolated, exactly as BATS-R-US interpolates `GradVar_DGB`.",
]),
("class MHDGrid", "MHDGrid: class docstring", [
    ("NFIELD = 5", "Five fields are stored per grid point, interleaved in one array: $n_e$, $\\partial_x n_e$, $\\partial_y n_e$, $\\partial_z n_e$, $T_e$. Interleaving means one memory fetch per cell corner returns all five values."),
]),
("def __init__(self, x, y, z, ne", "MHDGrid.__init__: input checks", [
    (("self.x = np.asarray", "self.z = np.asarray"), "Axes as float arrays (in $R_\\odot$)."),
    (("for a, name in", "raise ValueError(f\"axis"), "Each axis must be 1D, strictly increasing and have at least two points; otherwise interpolation is undefined."),
    (("shape = (", "raise ValueError(f\"ne has shape"), "The density must be shaped (nx, ny, nz) in `ij` ordering, the same order as the axes."),
    ("ne = np.where(np.isfinite(ne)", "NaN or infinite values become 0 and negative densities are clipped to 0. This guards against ghost-cell or undefined values in model output."),
    (("if te is None", "te = np.where"), "Temperature is optional (not needed for the simplistic mechanism); it defaults to 1 K. Non-finite values become 1 K."),
]),
("self.shape = shape", "MHDGrid.__init__: gradient and storage", [
    ("self.r_body = float", "Inner boundary radius: a ray stops when it reaches it (BATS-R-US `IsBody`)."),
    ("self.length_unit = float", "Metres per coordinate unit: $R_\\odot$ by default, $R_\\ast$ for stellar models. The tracer converts path lengths with it."),
    ("gx, gy, gz = np.gradient", "Density gradient with second-order central differences in the interior and second-order one-sided differences at the edges. Works for non-uniform axes. The unit is m$^{-3}$ per $R_\\odot$, matching positions in $R_\\odot$."),
    (("f = np.empty", "f[..., 4] = te"), "Fill the interleaved array, by default in float32 to halve memory (5 fields $\\times$ 4 bytes per point)."),
    ("self._flat = f.reshape", "Flatten to (Npoints, 5) so that a single integer index addresses a grid point."),
    ("del f, gx, gy, gz", "Free the temporary arrays."),
]),
("self.dx = np.diff(self.x)", "MHDGrid.__init__: helpers for fast lookup", [
    (("self.dx = np.diff", "self.dz = np.diff"), "Cell widths along each axis; used for interpolation weights and the local cell size."),
    (("self.lo = np.array", "self.hi = np.array"), "Corners of the domain box."),
    ("self._uniform", "If all three axes are uniform, locating a cell is a division (`floor`) rather than a binary search."),
    ("self._strides", "Flat-index strides for C-ordered (nx, ny, nz) arrays: index $=i\\,n_yn_z+j\\,n_z+k$."),
    (("offs = []", "self._corner_off"), "Pre-compute the 8 flat-index offsets of the cell corners $(i+d_i, j+d_j, k+d_k)$, $d\\in\\{0,1\\}$."),
]),
("# ------------------------------------------------------------------", "MHDGrid: properties", [
    ("def nbytes", "Memory used by the stored fields."),
    ("def __repr__", "One-line description printed by the driver (shape, extent, smallest cell, $r_\\text{body}$)."),
    ("def min_cell", "Smallest cell width over all axes."),
]),
("def inside(self, p)", "MHDGrid.inside()", [
    ("return np.all", "A point is inside if all three coordinates lie within $[\\text{lo},\\text{hi}]$. Vectorised over N points."),
]),
("def _locate(self, axis, d, u)", "MHDGrid._locate(): find the cell and the fractional position", [
    (("if self._uniform", "i = np.floor"), "Uniform axis: $i=\\lfloor(u-x_0)/\\Delta x\\rfloor$."),
    ("i = np.searchsorted", "Non-uniform axis: binary search, $O(\\log n)$ per point."),
    ("i = np.clip", "Clip to valid cells $[0,n-2]$; points outside are evaluated at the boundary value."),
    (("t = (u - axis[i])", "return i, np.clip"), "Fractional position $t\\in[0,1]$ inside the cell, used as the linear weight."),
]),
("def sample(self, p)", "MHDGrid.sample(): trilinear interpolation", [
    ("ins = self.inside(p)", "Domain flag returned to the tracer."),
    (("i, tx = self._locate", "k, tz = self._locate"), "Cell indices and weights along x, y, z."),
    ("base = i * self._strides", "Flat index of the lower corner of each point's cell."),
    (("out = np.zeros", "wz = (1 - tz, tz)"), "Accumulator for the 5 fields, and the linear weights $(1-t, t)$ along each axis."),
    (("for di in (0, 1):", "c += 1"), "Sum over the 8 corners: weight $w=w_x w_y w_z$ times the corner's 5 field values. Each step gathers N rows of the field table at once. This is the trilinear interpolation of BATS-R-US (`Weight_I` in `get_local_density`)."),
    ("cell = np.minimum", "Local cell size $\\min(\\Delta x,\\Delta y,\\Delta z)$: the target step `DeltaSNew`."),
    ("return np.maximum(out[:, 0]", "Return $n_e$ (clipped $\\ge0$), $\\nabla n_e$ (N,3), $T_e$, cell size and the inside flag."),
]),
("def box_intersection", "MHDGrid.box_intersection(): where a straight ray enters and leaves the box", [
    "Slab method: for each axis, the ray $\\vect o+t\\vect d$ is between the two faces for $t\\in[t_1,t_2]$; the intersection over the three axes is $[\\max t_\\text{min},\\min t_\\text{max}]$.",
    ("o = np.broadcast_to", "Same observer for all rays (broadcast)."),
    (("par = np.abs(d)", "dsafe"), "Avoid division by zero for directions parallel to a face."),
    (("t1 = (self.lo - o)", "tmax = np.maximum"), "Face crossings along each axis."),
    (("in_slab", "tmax = np.where(par"), "Parallel rays are either always inside that slab ($\\pm\\infty$) or never (empty interval)."),
    ("return tmin.max", "Entry and exit parameters; the ray misses if exit $<$ entry."),
]),
("# common API", "MHDGrid: common API", [
    ("domain_intersection = box_intersection", "Generic name used by `image.start_points`, so Cartesian and spherical grids are interchangeable."),
    ("def r_outer", "Largest radius inside the box; used to start parallel rays outside the model."),
    (("def to_npz", "te=self._flat"), "Save $n_e$, $T_e$ and the axes. `load_mhd('file.npz')` rebuilds the grid without re-reading the original model."),
]),
("class SphericalGrid", "SphericalGrid: class docstring", [
    "Used for PLUTO (and any) spherical output. Sampling is done in $(r,\\theta,\\phi)$ index space, so the stretched radial grid near the Sun is used as is.",
    ("NFIELD = 5", "Stored fields: $n_e$, $\\partial_r n_e$, $r^{-1}\\partial_\\theta n_e$, $(r\\sin\\theta)^{-1}\\partial_\\phi n_e$, $T_e$. These are the spherical components of the gradient vector, converted to Cartesian after interpolation."),
]),
("def __init__(self, r, theta, phi", "SphericalGrid.__init__: fields", [
    ("self.axisym = phi is None", "A single (or absent) $\\phi$ value means a 2.5D axisymmetric model."),
    ("self.phi = np.zeros(1)", "Store at least one $\\phi$ value so the array is always 3D."),
    (("ne = np.asarray(ne, float).reshape", "te = np.where"), "Reshape to (nr, nth, nph) and clean invalid values, as in `MHDGrid`."),
]),
("def edges(c, e)", "SphericalGrid.__init__: domain extent", [
    (("def edges(c, e)", "return np.concatenate"), "Use the given cell edges, or reconstruct them as mid-points between centres plus half a cell at each end."),
    ("self.theta_edges = np.clip", "Colatitude edges are limited to $[0,\\pi]$."),
    ("self.r_lo, self.r_hi", "Inner and outer radii of the domain."),
    ("self.full_theta", "Does the grid cover the whole colatitude range? If not, rays leaving the wedge stop."),
    (("span =", "self.full_phi"), "Full $2\\pi$ in longitude (or axisymmetric) means $\\phi$ is periodic."),
    ("self.r_body = float", "Rays stop at the inner boundary, by default the inner radial edge (usually 1 $R_\\odot$)."),
]),
("# gradient in spherical components", "SphericalGrid.__init__: gradient components", [
    (("R = self.r[:, None, None]", "S = np.maximum"), "Broadcast helpers $r$ and $\\sin\\theta$ ($\\ge10^{-6}$ to avoid division by zero at the poles)."),
    ("g_r = np.gradient", "$\\partial n_e/\\partial r$."),
    (("g_t = (np.gradient", "if nt > 1 else"), "$r^{-1}\\partial n_e/\\partial\\theta$."),
    (("if self.full_phi:", "g_p = np.gradient(pad"), "Periodic $\\phi$: pad one cell on each side with the wrapped values so that the central difference is correct at $\\phi=0$ and $2\\pi$."),
    ("g_p = np.gradient(ne, self.phi", "Non-periodic wedge: ordinary one-sided differences at the edges."),
    ("g_p = g_p / (R * S)", "$(r\\sin\\theta)^{-1}\\partial n_e/\\partial\\phi$."),
    ("g_p = np.zeros_like", "Axisymmetric: no $\\phi$ gradient."),
    (("f = np.empty", "self._flat"), "Interleaved storage as in `MHDGrid`."),
    (("self.dr = np.diff", "self.dph"), "Cell widths in $r$, $\\theta$, $\\phi$ (used for the step size)."),
    ("self._strides", "Flat-index strides for the (nr, nth, nph) array."),
]),
("def __repr__", "SphericalGrid: description and properties", [
    ("kind =", "Reports whether the model is axisymmetric or 3D, and its extent in degrees."),
    ("def min_cell", "Smallest of $\\Delta r$ and $r_\\text{lo}\\Delta\\theta$."),
]),
("def _sph(p)", "SphericalGrid._sph(): Cartesian to spherical", [
    ("r = np.sqrt", "$r=|\\vect p|$."),
    ("th = np.arccos", "Colatitude $\\theta=\\arccos(z/r)$ (clipped for round-off)."),
    ("ph = np.mod", "Longitude $\\phi=\\operatorname{atan2}(y,x)$ mapped to $[0,2\\pi)$."),
]),
("def _phi_rel", "SphericalGrid._phi_rel() and inside()", [
    ("return np.mod(ph - self.phi_edges[0]", "Longitude measured from the first edge, so a wedge starting at any angle can be tested with one comparison."),
    ("ok = (r >= self.r_lo)", "Inside the radial shell ..."),
    ("if not self.full_theta", "... and within the colatitude range, if limited ..."),
    ("if not self.full_phi", "... and within the longitude range, if limited."),
]),
("def _locate(c, u)", "SphericalGrid._locate()", [
    ("if n == 1", "A single cell (axisymmetric $\\phi$): index 0, weight 0."),
    (("i = np.clip(np.searchsorted", "t = np.clip"), "Binary search for the bracketing cell centres and the linear weight. The spherical axes are often stretched, so no uniform shortcut is used."),
]),
("def sample(self, p)", "SphericalGrid.sample(): locate in r and theta", [
    ("r, th, ph = self._sph(p)", "Spherical coordinates of the sample points."),
    (("ins = (r >= self.r_lo)", "ins &="), "Domain flag (radial shell, colatitude range)."),
    (("i, ti = self._locate", "j, tj = self._locate"), "Bracketing indices and weights in $r$ and $\\theta$."),
]),
("if nph == 1:", "SphericalGrid.sample(): longitude handling", [
    (("if nph == 1:", "tk = np.zeros"), "Axisymmetric: both $\\phi$ corners are the single cell, so the field does not depend on $\\phi$."),
    (("elif self.full_phi:", "tk = np.clip((phr - rel[k0])"), "Periodic longitude: the last cell centre is followed by the first one plus $2\\pi$. `k1 = (k0+1) % nph` wraps around, and the weight uses a $2\\pi$-shifted upper centre."),
    (("else:", "k1 = k0 + 1"), "Limited wedge: ordinary search; points outside the wedge are flagged."),
]),
("st = self._strides", "SphericalGrid.sample(): trilinear interpolation in index space", [
    (("for di, wi in", "out += w[:, None]"), "Weighted sum over the 8 corner cells. `np.minimum(..., size-1)` keeps the upper index valid when a point is beyond the last centre; the weight is then zero."),
]),
("ne, gr, gt, gp, te = out.T", "SphericalGrid.sample(): gradient to Cartesian, cell size", [
    (("ne, gr, gt, gp, te", "sph, cph"), "Unpack the interpolated fields and the trigonometric factors at each point."),
    (("grad = np.empty", "grad[:, 2] ="), "Transform $(g_r,g_\\theta,g_\\phi)$ to Cartesian with the unit vectors $\\hat r=(s_\\theta c_\\phi, s_\\theta s_\\phi, c_\\theta)$, $\\hat\\theta=(c_\\theta c_\\phi, c_\\theta s_\\phi, -s_\\theta)$, $\\hat\\phi=(-s_\\phi, c_\\phi, 0)$."),
    (("dr = self.dr[", "cell = np.minimum(dr, rdt)"), "Local cell size $\\min(\\Delta r, r\\Delta\\theta)$."),
    (("if nph > 1:", "cell = np.minimum(cell, np.maximum"), "Include $r\\sin\\theta\\,\\Delta\\phi$ in 3D, floored at a quarter of $r\\Delta\\theta$. The cells near the poles are tiny, and without the floor the step size would collapse there."),
    ("return np.maximum(ne, 0.0)", "Same return signature as `MHDGrid.sample`."),
]),
("def domain_intersection", "SphericalGrid.domain_intersection()", [
    (("b = np.einsum", "disc = b * b - c"), "Solve $|\\vect o+t\\vect d|=r_\\text{hi}$: $t=-b\\pm\\sqrt{b^2-c}$, with $b=\\vect o\\cdot\\vect d$ and $c=|\\vect o|^2-r_\\text{hi}^2$."),
    (("t0 = np.where", "t1 = np.where"), "Entry and exit; rays that miss the sphere get an empty interval."),
    ("box_intersection = domain_intersection", "Alias for symmetry with `MHDGrid`."),
]),
("def to_npz", "SphericalGrid.to_npz()", [
    "Saves everything needed to rebuild the grid, including the `geometry='spherical'` tag that `load_mhd` checks.",
]),
]))

# ============================================================================
MODULES.append(("raytrace.py", "raytrace.py --- the ray tracer",
r"""This is the core of the package: a port of `ray_path` in `ModRadioWaveImage.f90`. The
Fortran calls `ray_path` once per ray and step through the BATS-R-US particle
framework. Here one Python function advances \emph{all} rays together. Every quantity is
an array over the currently active rays, and the branches of the Fortran (`if` blocks per
ray) become boolean masks. Index conventions used throughout:
`act` holds the global indices of the rays still active; arrays such as `p`, `ne`, `eps`
are \emph{local} to `act`; `adv` holds local indices of rays that advance in this
iteration, and `ia = act[adv]` their global indices.""",
[
('"""Refractive', "Module docstring and imports", [
    "Lists the six stages of one step, which the blocks below implement in order.",
]),
("# Termination codes", "Ray status codes", [
    ("ACTIVE, ESCAPED", "Integer codes stored per ray: 0 active, 1 escaped, 2 hit body, 3 over-dense, 4 $\\varepsilon\\le0$, 5 optically thick, 6 step limit, 7 missed the grid."),
    ("STATUS_NAMES", "Names used in printed summaries."),
]),
("def trace_rays(", "trace_rays(): signature and docstring", [
    ("def trace_rays(", "`grid`, starting points, directions, frequency and emission model are required; the rest are keyword options. The defaults reproduce BATS-R-US: `tolerance=1e-2` (`Tolerance`), `step_min=1e-2` (`StepMin`)."),
    ("strict_batsrus", "Switch between the exact Fortran logic and the corrected default; explained in the docstring and in Section \\ref{sec:strict}."),
]),
("if transfer not in", "Initialisation of the ray state", [
    (("pos = np.array", "slope /= np.linalg.norm"), "Copies of positions $\\vect r$ (`State_VI(x_:z_)`) and unit directions $\\vect v$ (`State_VI(SlopeX_:SlopeZ_)`); the directions are re-normalised."),
    ("ncr = float(critical_density", "Critical density for this frequency (`NElectronSiCr`)."),
    ("tol2 = tolerance**2", "Squared tolerance (`Tolerance2`)."),
    ("rint2 =", "Squared integration-sphere radius plus 0.01, exactly as `rIntegration2` in Fortran; infinite if none."),
    ("rbody2", "Squared inner-boundary radius."),
    ("L = float(getattr(grid", "Metres per coordinate unit, from the grid: $R_\\odot$ for the Sun, $R_\\ast$ for a star with coordinates in stellar radii."),
    (("if hasattr(emission, \"kind\")", "emit = emission"), "Built-in emission models receive the length unit (needed by the simplistic mechanism, which is defined per normalised length); user functions are called as `f(ne, te, freq)`."),
    ("status = np.zeros", "Initial status (rays that miss the grid arrive here already flagged)."),
    (("intensity = np.zeros", "tau = np.zeros"), "Pixel intensity (`Intensity_I`) and accumulated optical depth."),
    ("nstep = np.zeros", "Step counter per ray."),
    (("dist2cr = np.zeros", "steep = np.zeros"), "Distance to the critical surface (`Dist2Cr_`) and the steep flag (`Steepness_`)."),
    ("ds = np.zeros", "Step size (`Ds_`)."),
]),
("act = np.flatnonzero(status == ACTIVE)", "Initial step size and path recording", [
    ("act = np.flatnonzero", "Indices of rays to trace."),
    (("if ds_init is None", "ds[act] = step_factor"), "Start with the local cell size at the entry point. The Fortran starts with `DeltaS = 1.0` $R_\\odot$, which is too large when the model box is small."),
    ("ds[:] = ds_init", "Or a user-given initial step."),
    (("paths = None", "slopes = {int(i)"), "Optionally record the position and direction of selected rays after every accepted step (used by `trace_paths` and the validation tests)."),
]),
("while act.size:", "Main loop: predictor and sampling", [
    ("while act.size:", "Iterate until every ray has ended. One pass is one attempted step for all active rays."),
    (("p = pos[act]", "half = 0.5 * d"), "Local copies: start position $\\vect r_0$ (`PositionHalfBack`), direction $\\vect v_0$, step $\\Delta s$ and $\\Delta s/2$ (`HalfDeltaS`)."),
    ("pm = p + s * half", "Predictor: move half a step, $\\vect r_{1/2}=\\vect r_0+\\vect v_0\\Delta s/2$. Same as the first statement of `ray_path`."),
    ("ne, gn, te, cell, inside = grid.sample", "Sample density, gradient, temperature and cell size at the mid point (`get_local_density`)."),
    ("dsnew = step_factor * cell", "Desired step `DeltaSNew`."),
    ("r2 = np.einsum", "$|\\vect r_{1/2}|^2$ for the body and sphere tests (`einsum` gives row-wise dot products)."),
    ("term = np.zeros", "Termination code decided in this iteration (0 = continue)."),
    ("term[r2 < rbody2] = HIT_BODY", "Mid point inside the body: the ray has reached the photosphere. This is tested first, because for a spherical grid the body is also outside the domain."),
    ("term[(term == 0) & (~inside", "Left the domain or the integration sphere: escaped."),
    ("term[(term == 0) & (ne >= ncr)]", "Over-dense plasma at the mid point. The Fortran stops the ray here (`NElectronSi >= NElectronSiCr`); the default mode may recover it below."),
]),
("# rays whose predictor point", "Rays that reach the photosphere", [
    "When the mid point is already inside the body, the piece of the step between $\\vect r_0$ and the surface would otherwise be lost. It is integrated as a straight segment.",
    (("ps, sv = p[hb]", "tt = np.clip"), "Distance to the sphere $r_\\text{body}$ along $\\vect v$: $t=-b-\\sqrt{b^2-c}$ with $b=\\vect r_0\\cdot\\vect v$ and $c=r_0^2-r_\\text{body}^2$, limited to $[0,\\Delta s]$."),
    (("ne_h, _, te_h", "k_h, B_h = emit"), "Emission at the mid point of that piece; the density is capped at $n_\\text{cr}$."),
    (("dl_h = tt * L", "tau[ih] += dtau_h"), "Add the contribution in the thin or full mode (same formulas as the main accumulation below)."),
]),
("eps = 1.0 - ne / ncr", "Dielectric permittivity and its gradient", [
    ("eps = 1.0 - ne / ncr", "$\\varepsilon_{1/2}=1-n_e/n_\\text{cr}$ (`DielPerm`)."),
    ("ge = -gn / ncr", "$\\nabla\\varepsilon=-\\nabla n_e/n_\\text{cr}$ in $R_\\odot^{-1}$ (`GradDielPerm_D`)."),
    ("ge2 = np.einsum", "$|\\nabla\\varepsilon|^2$ (`GradDielPerm2`)."),
    ("gs = np.einsum", "$\\nabla\\varepsilon\\cdot\\vect v_0$ (`GradEpsDotSlope`): negative when the ray moves towards denser plasma."),
    ("ehb = eps - gs * half", "Back-extrapolated permittivity at the step start, $\\varepsilon_0=\\varepsilon_{1/2}-\\nabla\\varepsilon\\cdot\\vect v_0\\,\\Delta s/2$ (`DielPermHalfBack`, Benkevitch Eq. 34)."),
    ("term[(term == 0) & (ehb <= 0.0)]", "$\\varepsilon_0\\le0$: the ray is already beyond the critical surface. The Fortran discards it (`mark_undefined`)."),
]),
("if not strict_batsrus and not no_refraction:", "Recovery of rays that crossed the critical surface (default mode)", [
    "Implements the fallback described by Benkevitch et al. (2010, Sec. 5.2) and Moschou et al. (2018, Sec. 2.2): return to the previous position, then reflect.",
    ("bad = (term == OVERDENSE)", "Rays that are over-dense at the mid point or have $\\varepsilon_0\\le0$."),
    (("back = bad & (d >= 4.0 * min_ds)", "ds[act[back]] = 0.25"), "First choice: retry from the same start point with a quarter of the step. The position is not changed, so the next iteration probes a point closer to $\\vect r_0$."),
    (("snell = bad & ~back", "steep[act[snell]] = False"), "If the step is already tiny, reflect the direction specularly about the gradient: $\\vect v\\leftarrow\\vect v-2(\\vect v\\cdot\\hat g)\\hat g$, $\\hat g=\\nabla\\varepsilon/|\\nabla\\varepsilon|$. This only applies if the ray moves into the gradient. The step is reset to `step_min` and the steep flag cleared."),
    (("term[back | snell] = 0", "ok_bad = back | snell"), "These rays stay active, but do not advance in this iteration."),
    ("ok = (term == 0) & ~ok_bad", "Rays that continue with the normal step logic."),
]),
("# ---- check_step", "Precision control (check_step)", [
    "Only for rays not marked steep, and only with refraction on (`UseNoRefraction` returns immediately in Fortran).",
    ("curv = (0.5 * half / ehb)**2", "$(\\kappa\\Delta s/2)^2=(\\Delta s/4\\varepsilon_0)^2(|\\nabla\\varepsilon|^2-(\\nabla\\varepsilon\\cdot\\vect v)^2)$, the squared turning angle over half a step. The bracket is the squared transverse gradient."),
    ("r_a = chk & (curv >= tol2)", "Too much bending for the tolerance (about 1/(2 Tol) = 50 steps per radian):"),
    ("newd[r_a] = 0.99", "... shrink the step by the factor needed to meet the tolerance, with a 1\\% margin (the Fortran formula)."),
    (("if not strict_batsrus:", "r_a &= d > min_ds"), "Default mode: never go below `min_ds`, and stop retrying once there. The Fortran has no lower bound."),
    ("r_b = chk & ~r_a & (3.0 * gs * half <= -ehb)", "Correctness criterion (Benkevitch Eq. 33): a linear prediction puts $\\varepsilon=0$ within about 1.5 steps."),
    ("d2c = ehb[r_b]", "Distance to the critical surface $d_\\text{cr}=\\varepsilon_0/|\\nabla\\varepsilon|$."),
    (("steep[idx] = True", "dist2cr[idx] = d2c"), "Mark the ray steep and remember $d_\\text{cr}$ (`Dist2Cr_`)."),
    ("newd[r_b] = np.maximum", "New step $\\max(\\tfrac12\\text{Tol}\\,d_\\text{cr}, \\text{StepMin})$."),
    (("retry = r_a | r_b", "ds[act[retry]] = newd[retry]"), "These rays redo the step with the smaller $\\Delta s$; their position stays at $\\vect r_0$ (in Fortran: `State_VI(x_:z_) = PositionHalfBack_D` and `RETURN`)."),
]),
("adv = np.flatnonzero(ok & ~retry)", "Advance: select rays and evaluate the emission", [
    ("adv = np.flatnonzero", "Rays that passed all checks advance now (local indices); `ia` are their global indices."),
    ("sa, pma, ha, da", "Direction, mid point, half step and step of the advancing rays."),
    ("kap, B = emit", "$\\kappa$ and $B$ at the mid point $\\vect r_{1/2}$ (`user_material_properties(State_V, ...)`)."),
    (("if kappa_over_mu", "kap = kap / np.sqrt"), "Optional refractive correction $\\kappa/\\mu$, with $\\mu=\\sqrt\\varepsilon$ floored at $10^{-3}$."),
]),
("if no_refraction:", "Advance: straight line of sight", [
    (("snew = sa", "seg = da"), "Direction unchanged; $\\vect r_1=\\vect r_{1/2}+\\vect v\\Delta s/2$; segment length $\\Delta s$."),
    (("ga, gsa, eha, ge2a", "refl |= (da < step_min)"), "With refraction: choose between the parabolic switch and a Boris step. The switch is used when the linear prediction says the ray would reach $\\varepsilon=0$ (`3*GradEpsDotSlope*HalfDeltaS <= -DielPermHalfBack`). The extra Fortran trigger `DeltaS < StepMin .and. GradEpsDotSlope < 0` is added only in strict mode."),
    (("snew = np.empty_like", "seg = da.copy()"), "Output arrays for both branches."),
]),
("# -- parabolic switch", "Advance: parabolic switch (reflection at the critical surface)", [
    ("lcos = -gsa[refl] / ge2a[refl]", "$L\\cos\\alpha=-\\nabla\\varepsilon\\cdot\\vect v/|\\nabla\\varepsilon|^2$, with $L=1/|\\nabla\\varepsilon|$ and $\\alpha$ the incidence angle."),
    ("proj = -lcos[:, None] * g", "Component of $\\vect v$ along $\\nabla\\varepsilon$: $\\vect v_\\parallel=-L\\cos\\alpha\\,\\nabla\\varepsilon$ (`ProjSlopeOnMinusGradEps_D`)."),
    ("stepy = sr - proj", "Transverse component $\\vect v-\\vect v_\\parallel$, of length $\\sin\\alpha$."),
    ("snew[refl] = sr - 2.0 * proj", "Reflected direction: the parallel component is reversed (Snell's law at a mirror)."),
    ("stepy = 4.0 * stepy", "Scale to the chord of the parabola: $|\\Delta\\vect r_\\perp|=4\\sin\\alpha\\cos\\alpha\\,\\varepsilon_0 L$."),
    ("pnew[refl] = p[adv][refl] + stepy", "Jump from $\\vect r_0$ to the symmetric point on the other branch."),
    ("stepx = (eha[refl] * lcos**2)", "Depth of the parabola vertex along $\\nabla\\varepsilon$: $\\varepsilon_0(L\\cos\\alpha)^2|\\nabla\\varepsilon|$."),
    ("seg[refl] = np.sqrt", "Path length assigned to the arc, $\\sqrt{|2\\Delta\\vect r_\\parallel|^2+|\\Delta\\vect r_\\perp|^2}$ (`ParabLen`)."),
]),
("# -- Boris step", "Advance: Boris (CYLRAD) step", [
    ("coef = 0.5 * ha[bo] / eps[adv][bo]", "$\\Delta s/(4\\varepsilon_{1/2})$ (`Coef`)."),
    ("rel = coef[:, None] * ga[bo]", "$\\vect a=\\nabla\\varepsilon\\,\\Delta s/(4\\varepsilon)=(\\nabla\\mu/\\mu)\\,\\Delta s/2$ (`RelGradRefrInx_D`)."),
    (("om = np.cross(rel, sb)", "s1 = sb + np.cross(sb, om)"), "First estimate: $\\vect\\Omega_0=\\vect a\\times\\vect v_0$, $\\vect v'=\\vect v_0+\\vect v_0\\times\\vect\\Omega_0$."),
    (("om = np.cross(rel, s1)", "s1 = sb + np.cross(sb, om)"), "Rotation vector at the half step $\\vect\\Omega_{1/2}=\\vect a\\times\\vect v'$, and $\\vect v''=\\vect v_0+\\vect v_0\\times\\vect\\Omega_{1/2}$."),
    (("c2 = 2.0 /", "sn = sb + c2"), "Exact-length rotation $\\vect v_1=\\vect v_0+\\frac{2}{1+\\Omega^2}\\vect v''\\times\\vect\\Omega_{1/2}$, so $|\\vect v_1|=1$."),
    (("snew[bo] = sn", "pnew[bo] = pma[bo]"), "Corrector: $\\vect r_1=\\vect r_{1/2}+\\vect v_1\\Delta s/2$."),
]),
("# ---- clip the last segment", "Clip a step that ends inside the photosphere", [
    ("into_body = r2e < rbody2", "Steps whose end point lies inside $r_\\text{body}$."),
    (("ps, dd_ =", "frac[into_body]"), "Solve $|\\vect r_0+t(\\vect r_1-\\vect r_0)|=r_\\text{body}$ for the first root $t\\in[0,1]$ and keep only that fraction of the segment. The Fortran counts the whole step; the difference matters at disk centre at high frequency."),
]),
("# ---- accumulate the intensity", "Accumulate the emission", [
    ("dl = seg * frac * L", "Segment length in metres: the step in coordinate units times the length unit ($R_\\odot$ or $R_\\ast$)."),
    ("intensity[ia] += kap * B * dl", "Optically thin, as in BATS-R-US: $I\\mathrel{+}=\\kappa B\\Delta\\ell$ (with `FrequencySi_W(WaveLast_)=1`)."),
    (("dtau = kap * dl", "tau[ia] += dtau"), "Full transfer: add $B e^{-\\tau}(1-e^{-\\Delta\\tau})$, the exact solution for constant $\\kappa$, $B$ on the segment, then update $\\tau$. `-expm1(-x)` is $1-e^{-x}$, accurate for small $x$."),
    (("pos[ia] = pnew", "slope[ia] = snew"), "Store the new state."),
]),
("# ---- exclude rays which left", "End-of-step tests", [
    ("gone =", "The ray left the domain (and not by entering the body) or the integration sphere: status escaped."),
    ("status[ia[into_body & ~gone]] = HIT_BODY", "The ray ended on the photosphere."),
    ("status[ia[(~gone) & (tau[ia] > tau_max)]] = OPAQUE", "Full transfer: once $\\tau>\\tau_\\text{max}$ (30 by default; $e^{-30}\\approx10^{-13}$) nothing more can be seen, so the ray stops early. This saves time."),
]),
("# ---- get_new_step_size", "Step regrowth (get_new_step_size)", [
    ("cond = ~steep[ia]", "Regrow the step for gentle rays, and for steep rays that are now farther from the critical surface than when they were marked ($\\varepsilon_0>d_\\text{cr}|\\nabla\\varepsilon|$)."),
    ("cond[:] = True", "Without refraction, always regrow."),
    ("steep[ic] = False", "Those rays are gentle again."),
    ("ds[ic] = np.maximum", "Non-linear update $\\Delta s'=\\max(2-\\Delta s/\\Delta s_\\text{new},1)\\min(\\Delta s_\\text{new},\\Delta s)$. Small steps roughly double each time and saturate smoothly at $\\Delta s_\\text{new}$, which is reached after a few steps."),
]),
("if paths is not None:", "Path recording", [
    ("for i in ia[rec_mask[ia]]", "Append the new position and direction of the recorded rays."),
]),
("tmask = (term > 0) & ~ok_bad", "Bookkeeping at the end of the iteration", [
    ("status[act[tmask]] = term[tmask]", "Store the termination codes decided at the mid point."),
    ("nstep[act] += 1", "Count the iteration (including retries)."),
    (("over =", "status[act[over]] = MAX_STEPS"), "Safety stop for runaway rays."),
    ("act = act[status[act] == ACTIVE]", "Keep only the rays that are still active; the loop ends when none are left."),
]),
('out = {"intensity"', "Return value", [
    ("out = {", "Intensity [W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$], optical depth, status and step count per ray."),
    (("if paths is not None", "out[\"slopes\"]"), "Recorded trajectories, converted to (M,3) arrays."),
]),
]))

# ============================================================================
MODULES.append(("image.py", "image.py --- image set-up and driver",
r"""Builds the rays for an image (port of `write_plot_radiowave`), runs the tracer for
each frequency (optionally in parallel), and wraps the result in a `RadioImage`.""",
[
('"""Synthetic radio image driver', "Module docstring and imports", [
    "Summarises the image geometry, which is identical to BATS-R-US.",
    ("from .raytrace import", "The tracer and the status codes."),
]),
("def parse_frequencies", "parse_frequencies()", [
    "Python version of `parse_freq_string`: accepts `'1500kHz, 11MHz, 42.7MHz'`, a number, or a list of numbers in Hz.",
    (("if np.isscalar(spec)", "for v in spec])"), "Numbers are taken as Hz; a list may mix numbers [Hz] and strings such as '80MHz'."),
    ("for num, unit in re.findall", "Regular expression: a number (Fortran `d` exponents allowed) followed by an optional unit."),
    (("unit = (unit or \"Hz\")", "raise ValueError"), "Units are case-insensitive. A number without a unit is in Hz."),
    ("out.append(abs(", "Negative values are made positive, as in the Fortran."),
]),
("def freq_label", "freq_label()", [
    ("for lim, u in", "Readable label such as '150.00 MHz' for titles and file headers."),
]),
("def image_geometry", "image_geometry(): pixels and ray directions", [
    (("obs = np.asarray", "normal = obs / dist"), "Observer position and $\\hat n=\\vect O/|\\vect O|$ (`Normal_D`)."),
    ("tau = np.cross(up, normal)", "$\\hat\\tau=\\hat z\\times\\hat n$ (`Tau_D`), the image X axis."),
    (("if np.linalg.norm(tau) < 1e-12", "tau = np.cross(np.array"), "Guard: if the observer lies on the up axis the cross product vanishes, so $\\hat y$ is used instead."),
    ("xi = np.cross(normal, tau)", "$\\hat\\xi=\\hat n\\times\\hat\\tau$ (`Xi_D`), the image Y axis."),
    (("dxp = x_size / nx", "yc = -0.5 * y_size"), "Pixel sizes and pixel-centre coordinates $X_i=X_\\text{lo}+(i-\\tfrac12)\\Delta X$."),
    (("X, Y = np.meshgrid", "pix = X.reshape"), "3D positions of all pixel centres, ordered with X fastest (ray index $= j\\,n_x+i$ as in Fortran)."),
    (("if parallel:", "d = np.broadcast_to"), "Distant observer (stars, `parallel=True`): all rays are parallel, pointing along $-\\hat n$."),
    (("d = pix - obs", "d /= np.linalg.norm"), "Otherwise: unit directions from the observer to the pixels (`SlopeUnscaled_D` normalised), as in BATS-R-US."),
    ("return xc, yc, d, tau, xi, normal, pix", "The 3D pixel positions are also returned; the parallel projection uses them as ray origins."),
]),
("def start_points", "start_points(): where each ray starts", [
    (("if obs.ndim == 1", "od = np.einsum"), "One observer for all rays (perspective), or one origin per ray (parallel projection)."),
    (("disc = r_integration", "t_sph = -od"), "Entry into the integration sphere: the smaller root of $|\\vect O+t\\vect d|=r_\\text{int}$ (`ObservToIntSphereDist`)."),
    ("t_in, t_out = grid.domain_intersection", "Entry into and exit from the model domain (box or outer sphere)."),
    ("t0 = np.maximum", "Start at the later of the two entries, and never behind the observer."),
    (("t_sph_out", "t1 = np.minimum"), "End of the useful interval."),
    ("status0[(disc <= 0) | (t1 <= t0)] = MISSED", "Rays that never pass through the model inside the sphere are flagged and skipped."),
    (("span =", "t0 = t0 + np.minimum"), "Nudge the start point slightly inside, so that round-off does not put it just outside the box face."),
]),
("_G = {}", "Parallel worker", [
    ("_G = {}", "Module-level storage for the shared data. With the `fork` start method the worker processes inherit it without copying or pickling the grid."),
    ("def _worker", "Traces one chunk of ray indices and returns the results with the indices."),
]),
("class RadioImage", "RadioImage: container", [
    ("def __init__", "Stores intensity, $\\tau$, status, step counts (all (nf, ny, nx)), pixel axes, frequencies, observer and the settings."),
    (("self.distance = float(", "self.length_unit = float"), "Observer distance in coordinate units (the true distance for a star, otherwise $|\\vect O|$) and metres per coordinate unit. Together they set the flux density and the angular scale."),
]),
("def brightness_temperature", "RadioImage: derived quantities", [
    ("return self.intensity * C_LIGHT**2", "$T_b=Ic^2/(2k_B\\nu^2)$."),
    ("def pixel_solid_angle", "$\\Delta\\Omega=\\Delta X\\Delta Y/D^2$: pixel area over distance squared, both in coordinate units."),
    ("def flux_density", "$F=\\sum I\\,\\Delta\\Omega$ per frequency, in W m$^{-2}$ Hz$^{-1}$."),
    ("def flux_density_sfu", "The same in solar flux units."),
    ("def flux_density_mjy", "The same in mJy, the natural unit for stars."),
    ("def distance_pc", "Observer distance in parsec."),
]),
("def summary", "RadioImage.summary()", [
    ("counts = ", "For each frequency: maximum intensity and $T_b$, flux in SI and Jy, and how many rays ended in each way."),
]),
("def save_npz", "Output methods", [
    ("np.savez_compressed", "NPZ with all arrays, including $T_b$ and the settings (as text)."),
    (("def save_tecplot", "np.savetxt"), "ASCII Tecplot ordered zone with variables X, Y and one column per frequency, the same layout as the BATS-R-US radio image file."),
    (("def save_fits", "write_fits(path, data, h)"), "FITS cube with a simple linear WCS in $R_\\odot$ and the frequencies and observer in the header. It is written with astropy if installed, otherwise with the built-in writer in `fitsio.py`."),
]),
("def plot(self", "RadioImage.plot()", [
    ("matplotlib.use(\"Agg\")", "Non-interactive backend when writing a file (works on clusters without a display)."),
    (("ncols = ncols or", "fig, axes"), "One panel per frequency, up to three per row."),
    ("ext = [", "Image extent at the pixel edges, for `imshow`."),
    (("if log and pos.size", "cmap=cmap, norm=norm)"), "Logarithmic colour scale over four decades by default; pixels with zero intensity are left blank."),
    (("if show_limb", "ax.plot(np.cos(t)"), "Dashed circle marking the photospheric limb ($r=1$)."),
]),
("def make_radio_image", "make_radio_image(): the main entry point", [
    ("freqs = parse_frequencies", "Frequency list in Hz."),
    ("em = emission if callable", "A string selects the built-in `EmissionModel`; a function is used as given."),
    ("xc, yc, dirs", "Pixel axes and ray directions."),
    (("diag =", "r_integration = float(math.ceil"), "`rIntegration = ceiling(max(ImagePlaneDiagRadius+1.0, 5.0))`, as in Fortran."),
    (("L = float(getattr(grid", "parallel = distance_pc"), "Length unit of the grid; `distance_pc` switches to the distant-observer (parallel-ray) projection."),
    (("if parallel:", "pos, status0 = start_points(grid, origins"), "Parallel projection: the distance in coordinate units sets the flux. Each ray starts far outside the model on the observer's side of its pixel, and then enters the integration sphere / domain like any other ray."),
    (("else:", "pos, status0 = start_points(grid, observer"), "Perspective (Sun seen from the Earth): all rays start at the observer."),
    (("if verbose:", "rays never enter the grid box"), "Report the set-up."),
]),
("kw = dict(", "make_radio_image(): tracing", [
    ("kw = dict(", "Options passed to `trace_rays` (plus any advanced keywords)."),
    (("I = np.zeros", "NS = np.zeros"), "Result arrays (frequency, ray)."),
    ("use_pool", "Parallel only if more than one worker is requested and `fork` is available (Linux, macOS)."),
    (("_G.update", "chunks = "), "Share the data and split the rays into $4\\times$ `n_workers` interleaved chunks (rays $c, c+N, c+2N, \\ldots$). Rays near the disk take more steps than off-limb rays, so interleaving balances the load."),
    (("with mp.get_context", "I[k, idx]"), "Run the chunks and scatter the results back by index."),
    (("r = trace_rays", "I[k], T[k], S[k], NS[k]"), "Serial path: all rays in one call."),
]),
("shp = (len(freqs)", "make_radio_image(): result", [
    ("shp =", "Reshape (ray) to (ny, nx): with X fastest, row $j$ is image Y."),
    ("meta = dict(", "Record the settings with the image."),
    ("img = RadioImage(", "Build and return the result object, printing the summary if `verbose`."),
]),
("def trace_paths", "trace_paths(): individual trajectories", [
    "Traces rays through arbitrary image-plane points and records their paths. It uses the same geometry, start points and tracer as an image, so the paths are exactly those that make up the image.",
    (("pix = xp[:, None] * tau_v", "d /= np.linalg.norm"), "Directions to the requested image-plane points."),
    ("record_ids=np.arange", "Record every ray."),
]),
]))

# ============================================================================
MODULES.append(("readers.py", "readers.py --- BATS-R-US and generic input",
r"""Turns model files into dictionaries of point variables, works out $n_e$ and $T_e$
with the right units, arranges the points on a grid (or resamples them), and returns an
`MHDGrid`. `load_mhd` is the single entry point and also dispatches to the PLUTO
reader.""",
[
('"""Readers that turn', "Module docstring and imports", [
    "Lists the supported formats. Every reader returns `{name: array}`, which `build_grid` converts.",
]),
("def _base(name)", "Variable-name helpers", [
    ("def _base(name)", "Base name in lower case: `'Rho [g/cm^3]'` becomes `'rho'`."),
    ("def _unit(name)", "Unit in square brackets, lower case without spaces: `'g/cm^3'`."),
    ("def _find(vars_, candidates)", "First variable whose base name is in the candidate list."),
]),
("_MASS_UNITS", "Unit tables", [
    ("_MASS_UNITS", "Mass-density units and their factor to kg m$^{-3}$."),
    ("_NUM_UNITS", "Number-density units and their factor to m$^{-3}$. BATS-R-US 'amu/cc' and 'Mp/cc' are number densities."),
    ("_PRES_UNITS", "Pressure units and their factor to Pa (1 dyn cm$^{-2}$ = 0.1 Pa)."),
]),
("def _density_to_ne", "_density_to_ne(): density in any unit to n_e", [
    (("if u in _MASS_UNITS", "return vals * _MASS_UNITS"), "Mass density: $n_e=\\rho/(\\mu m_p)$."),
    (("if u in _NUM_UNITS", "return vals * _NUM_UNITS"), "Number density: unit conversion only."),
    (("if r is not None", "ref = np.nanmedian(np.abs(vals))"), "Unknown unit: take the median value at $r=1.05$--$1.3\\,R_\\odot$ as reference."),
    (("guesses =", "best = min"), "Try g/cm$^3$, kg/m$^3$, cm$^{-3}$ and m$^{-3}$ (or only the mass or number options if the name says which) and pick the one that gives $n_e$ closest to $10^{14.5}$ m$^{-3}$, a typical low-corona value. The choice is printed, so you can check it."),
]),
("def _pressure_si", "_pressure_si()", [
    ("return vals * 0.1", "Without a recognised unit, the BATS-R-US dimensional default dyn cm$^{-2}$ is assumed."),
]),
("def derive_ne_te", "derive_ne_te(): electron density", [
    ("r = np.sqrt", "Radius of every point, for the unit guess and the printed check."),
    ("dk = density_var or _find", "Density variable: user choice or the first of ne, n\\_e, rho, n, dens, density."),
    (("unit = density_unit", "unit = vars_[\"_units\"]"), "Unit: from the user, from the name, or from the IDL header units."),
    (("kind = None", "kind = \"number\""), "If there is no unit, the name decides whether it is a mass or a number density."),
    ("ne, how = _density_to_ne", "Convert to m$^{-3}$ and remember how, for the report."),
]),
("units = vars_.get", "derive_ne_te(): electron temperature", [
    (("tk = te_var or _find", "info[\"te\"] = f\"{tk} [K]\""), "1st choice: a Te variable in K."),
    (("pk = pe_var or _find", "te = pe / (np.maximum"), "2nd: electron pressure, $T_e=P_e/(n_ek_B)$ (AWSoM writes Pe)."),
    (("pk = p_var or _find", "te = p / (2.0"), "3rd: total pressure, $T=P/(2n_ek_B)$, i.e. $T_e=T_i$ in a hydrogen plasma."),
    ("tk = _find(vars_, [\"t\"", "4th: a generic temperature variable."),
    (("if te_const is None", "te = np.full_like"), "Last: a constant given by the user, otherwise an error."),
    (("if verbose:", "print(f\"radiosynth: n_e from"), "Print what was used and the median $n_e$, $T_e$ near $r=1.1\\,R_\\odot$. This is the best check that the units are right."),
]),
("def _unique_axis", "_unique_axis()", [
    (("scale =", "uq, inv = np.unique"), "Round coordinates to a relative tolerance of $10^{-6}$ and find the distinct values. This copes with single-precision round-off in the file."),
    (("ax = np.zeros", "ax /= np.bincount"), "Each axis value is the mean of the coordinates in its group; `inv` maps every point to its axis index."),
]),
("def points_to_grid", "points_to_grid(): tensor grid or resampling", [
    (("ax_x, ix", "n = x.size"), "Distinct coordinate values along each axis."),
    (("if ax_x.size * ax_y.size", "return ax_x, ax_y, ax_z, out"), "If the number of points equals $n_xn_yn_z$ and no two points share a cell, the data already form a tensor grid. They are placed by index, so any point order works (Tecplot I-fastest, shuffled, etc.)."),
]),
("# ---- resample scattered data", "points_to_grid(): resampling scattered points", [
    (("shape = resample_shape", "gz = np.linspace"), "Target uniform grid (default $160^3$) over the given or full box."),
    (("pts = np.column_stack", "G = np.stack"), "Source points and target points."),
    (("vals = []", "vals = np.column_stack"), "The density is interpolated as $\\log n_e$: it varies over many orders of magnitude, and linear interpolation of the logarithm keeps it positive and smooth."),
    (("if method == \"nearest\"", "res = vals[idx]"), "Fast nearest-neighbour lookup with a k-d tree."),
    (("LinearNDInterpolator", "res[bad] = vals[idx]"), "Linear interpolation on the Delaunay triangulation; target points outside the convex hull fall back to the nearest neighbour."),
    (("out = {}", "out[k] = (np.exp(a)"), "Undo the logarithm and reshape to 3D."),
]),
("def build_grid", "build_grid() and grid_from_arrays()", [
    ("ne, te, info = derive_ne_te", "Physical fields at the points."),
    ("x, y, z, f = points_to_grid", "Arrange or resample."),
    ("g = MHDGrid", "Build the grid and attach the conversion info. With `r_star` (in $R_\\odot$) the coordinates are taken to be in stellar radii, as in BATS-R-US stellar runs, and the length unit is $R_\\ast$."),
    ("def grid_from_arrays", "Convenience wrapper around `MHDGrid`."),
]),
("def _standardise_xyz", "_standardise_xyz()", [
    ("for c in (\"x\", \"y\", \"z\")", "Make sure the coordinates are available as plain 'x', 'y', 'z', whatever their original names (for example 'X [R]')."),
]),
("def read_tecplot_ascii", "read_tecplot_ascii()", [
    (("mvar = re.search", "names = [n.strip()"), "Variable names from the VARIABLES line, quoted or not, possibly over several lines."),
    (("zones = [m for m", "ztext = text"), "Select the requested zone."),
    (("lines = ztext.split", "header = \" \".join"), "The zone header runs until the first line that starts with a number."),
    (("def geti(key)", "E = geti"), "Read I, J, K (ordered zone) or N, E (finite-element zone) from the header."),
    ("block = bool", "BLOCK packing stores one variable after another; POINT packing stores one point after another."),
    ("npts = N if N", "Number of points."),
    (("body = ", "arr = np.array(body.split()"), "Parse all numbers in one go (Fortran 'D' exponents accepted). The connectivity list of FE zones, which follows the data, is ignored."),
    ("data = data.reshape(nv, npts) if block", "Arrange as (variable, point) for either packing."),
]),
("def read_pytecplot", "read_pytecplot()", [
    ("ds = tp.data.load_tecplot", "The same call as in the original script `Radio_Emsssion_Calculation_batsrus.py`. Every variable of the zone is read as a NumPy array."),
]),
("def _read_records", "_read_records(): Fortran unformatted files", [
    "Fortran sequential files wrap every record in 4-byte length markers. The endianness is found from the first record, whose leading and trailing markers must agree.",
    (("(L,) = struct.unpack(endian", "pos += 8 + L"), "Split the file into records."),
]),
("def read_batsrus_idl", "read_batsrus_idl(): binary IDL files", [
    (("first = fh.read(4)", "binary = True"), "Binary if the first 4 bytes form a plausible record length; ASCII files start with text."),
    ("head = recs[0]", "Record 1: header line (often the units)."),
    (("r2 = recs[1]", "ndim, npar, nvar"), "Record 2: nstep, time, ndim, npar, nvar. Its length gives the real size (4 bytes for real4, 8 for real8)."),
    ("gencoord = ndim < 0", "Negative ndim means generalised (unstructured) coordinates."),
    ("n_d = struct.unpack", "Record 3: grid dimensions."),
    (("if npar > 0:", "ir += 1"), "Record 4 (if present): equation parameters (gamma, rbody, ...)."),
    ("names = recs[ir]", "Next: names of coordinates, variables and parameters."),
    (("coord = np.frombuffer", "coord.reshape((npts, ndim), order=\"F\")"), "Coordinates, stored in Fortran order as (point, dimension)."),
    (("for _ in range(nvar)", "vals.append"), "One record per variable."),
]),
("    else:", "read_batsrus_idl(): ASCII files and units", [
    (("head = fh.readline()", "arr = np.loadtxt"), "ASCII layout: the same header items on lines, then one row per point."),
    (("cnames =", "out[vn] = vals[c]"), "Name the coordinate and variable arrays."),
    (("toks = head.split()", "units[n] ="), "If the header line lists one unit per coordinate and variable (for example 'R R R g/cm3 km/s ...'), attach them as `_units`."),
    ("out[\"_meta\"]", "Keep the header information."),
    (("if ndim != 3", "raise ValueError(f\"non-Cartesian"), "Only 3D Cartesian coordinates are accepted here; spherical models go through `SphericalGrid`."),
]),
("def load_mhd", "load_mhd(): the general entry point", [
    (("if reader is None and (ext == \".vtk\"", "reader = \"pluto\""), "A directory containing grid.out, or a .vtk file, is PLUTO output."),
    (("if reader == \"pluto\"", "if k in keep and v is not None})"), "Pass only the PLUTO-relevant keywords to `load_pluto`."),
    (("build_kw = {k: v", "build_kw.pop(\"r_body\""), "Drop the PLUTO keywords and unset values for the other readers."),
    ("reader = {\".dat\"", "Choose the reader from the file extension."),
    (("if reader == \"tecplot\"", "v = read_batsrus_idl"), "Call the reader."),
    (("elif reader == \"npz\"", "v = {k: z[k]"), "A saved grid is rebuilt directly (spherical or Cartesian); any other npz is treated as point data."),
    (("v = {k: (np.ravel(a)", "return build_grid"), "Flatten the arrays and build the grid."),
]),
]))

# ============================================================================
MODULES.append(("pluto.py", "pluto.py --- PLUTO input",
r"""Reads PLUTO binary (`.dbl`/`.flt`) and legacy VTK output, converts code units to
physical units using `definitions.h`, and returns a `SphericalGrid` or `MHDGrid`.""",
[
('"""Reader for PLUTO', "Module docstring", [
    "Documents the supported formats, the unit conversion and the plasma model ($n_e$, $T$).",
]),
("PLUTO_CONST = dict(", "PLUTO constants and default units", [
    ("PLUTO_CONST = dict(", "The physical constants of PLUTO's `pluto.h` (cgs), so that `definitions.h` expressions such as `CONST_mp*1.e8` evaluate exactly as in PLUTO."),
    ("PLUTO_DEFAULT_UNITS", "PLUTO's defaults when `definitions.h` does not set them: $U_\\rho=m_p$, $U_L=$ 1 AU, $U_v=$ 1 km/s."),
]),
("def parse_definitions", "parse_definitions()", [
    (("env = dict(PLUTO_CONST)", "env.update"), "Evaluation namespace: the constants plus a few math functions."),
    ("m = re.match", "Match `#define NAME expression` lines (a trailing C comment is ignored)."),
    (("val = eval(", "continue"), "Evaluate the expression without Python built-ins. Non-numeric defines (such as `PHYSICS MHD`) fail and are skipped."),
    (("env[name] = float(val)", "out[name]"), "Numeric results are stored and become available to later defines."),
]),
("def read_grid_out", "read_grid_out()", [
    (("while k < len(lines) and lines[k]", "geom = m.group(1)"), "Skip the comment header, picking up the `GEOMETRY` line."),
    (("n = int(tok[0])", "k += n"), "For each direction: the number of cells, then one line per cell."),
    (("if rows.shape[1] >= 5", "xl, xc, xr = rows"), "PLUTO 3 layout: i, left, centre, right, width."),
    (("xl, xr = rows[:, 1]", "xc = 0.5"), "PLUTO 4 layout: i, left, right; the centre is the midpoint."),
    ("while len(dims) < 3", "Fill missing dimensions of 1D/2D runs with a single cell."),
]),
("def _read_out_file", "_read_out_file()", [
    (("rows =", "raise ValueError"), "Pick the line for the requested output number, or the last line by default."),
    ("return int(row[0])", "Output number, time, single/multiple files, endianness and the variable list."),
]),
("def _stag_shape", "_stag_shape()", [
    "Staggered magnetic-field components (e.g. `Bx1s`) have one more value along their direction. Their size is needed to skip over them in a single-file output.",
]),
("def read_pluto_binary", "read_pluto_binary()", [
    ("fmt = \"dbl\" if", "Use .dbl if a dbl.out exists, otherwise .flt."),
    ("nout, time, mode, endian, names", "Description of the requested output."),
    ("geom, dims = read_grid_out", "Geometry and cell edges."),
    ("dt = np.dtype(", "Data type: float64 or float32, little or big endian."),
    (("if mode == \"single_file\"", "off += cnt"), "Single file: the variables follow each other. Seek to each wanted variable using the sizes of the preceding ones, so only rho, prs (and T/Te) are read."),
    ("a.reshape(shp[::-1]).transpose(2, 1, 0)", "PLUTO writes $x_1$ fastest. Read as (n3, n2, n1) and transpose to (n1, n2, n3)."),
    (("else:", "out[name] = a.reshape"), "Multiple files: one file per variable, `name.NNNN.dbl`."),
]),
("def read_pluto_vtk", "read_pluto_vtk(): header and helpers", [
    (("def line()", "return s"), "Read one text line from the byte buffer."),
    (("def arr(count, typ)", "return a"), "Read a big-endian binary array (VTK legacy binary is always big-endian)."),
    (("line(); line()", "kind = line()"), "Skip the version and title lines, check BINARY, read the dataset type."),
]),
("    while pos < len(data):", "read_pluto_vtk(): sections", [
    (("if key == \"FIELD\"", "time = float(v[0])"), "Field data: PLUTO stores the time here."),
    ("elif key == \"DIMENSIONS\"", "Number of grid nodes (cells + 1) per direction."),
    ("elif key in (\"X_COORDINATES\"", "Rectilinear grid: node coordinates along each axis."),
    ("elif key == \"POINTS\"", "Structured grid: Cartesian coordinates of every node."),
    (("elif key == \"SCALARS\"", "out[name] = a.reshape"), "Cell data: read the wanted variables, skip the others."),
    (("elif key == \"VECTORS\"", "arr(3 * int"), "Vector data are skipped."),
]),
("if kind == \"RECTILINEAR_GRID\"", "read_pluto_vtk(): geometry", [
    (("geom = \"CARTESIAN\"", "dd.append"), "Cartesian: node coordinates give the cell edges directly."),
    (("geom = \"SPHERICAL\"", "dd = [(e[:-1]"), "Spherical: convert the node positions back to $r$, $\\theta$, $\\phi$ and read the edges along one grid line in each direction."),
]),
("def _import_pypluto", "_import_pypluto(): find a pyPLUTO module", [
    "Lets radiosynth use the pyPLUTO reader that comes with PLUTO, as an alternative to the built-in reader.",
    (("if pypluto_path:", "return importlib.import_module(\"pyPLUTO\")"), "If a directory (or the path of pyPLUTO.py) is given, put it first on `sys.path` and import from there."),
    (("try:", "return importlib.import_module(\"pyPLUTO\")"), "Otherwise use an installed pyPLUTO (classic or the new PyPLUTO package) ..."),
    ("except ImportError", "... and if none is installed, the copy bundled in `radiosynth/contrib/pyPLUTO.py` (the classic `pload` version, with the multiple-files fix)."),
]),
("def _to_n123", "_to_n123(): put arrays in (n1, n2, n3) order", [
    "Different readers return arrays in different orders: classic pyPLUTO returns (n1, n2, n3) for 3D and (n1, n2) for 2D, while other tools return C order (n3, n2, n1). This helper accepts any of these and returns (n1, n2, n3).",
    ("if a.ndim == 3 and a.shape == tuple(n[::-1])", "Reversed (C-order) 3D array: transpose."),
    (("sq = tuple(", "return a.T.reshape(n)"), "2D/1D outputs: restore the singleton dimensions."),
]),
("def read_pluto_pypluto", "read_pluto_pypluto(): read through pyPLUTO", [
    (("pp = _import_pypluto", "geom, _ = read_grid_out"), "Import pyPLUTO, choose dbl/flt, find the last output number in dbl.out/flt.out if none is given, and read the geometry from grid.out (classic pyPLUTO does not report it)."),
    (("if hasattr(pp, \"pload\")", "time = float(getattr(D, \"time\""), "Classic API: `pyPLUTO.pload(ns, w_dir=..., datatype='double'|'float')`. The variables become attributes (`D.rho`, `D.prs`, ...), with the list in `D.vars` and the time in `D.time`."),
    (("elif hasattr(pp, \"Load\")", "time = float(np.atleast_1d"), "Newer PyPLUTO API: `pyPLUTO.Load(nout, path=...)`, which also gives the geometry in `D.geom`."),
    (("x1r, x2r, x3r =", "x2r = np.array"), "Cell edges `x1r`, `x2r`, `x3r` (pyPLUTO attributes). Missing dimensions of 2D runs get a single cell (the full $\\phi$ or $\\theta$ range for spherical runs)."),
    (("dims =", "n = tuple"), "The same (left edge, right edge, centre) triplets as `read_grid_out`."),
    (("for v in want:", "break"), "Copy rho, prs and a temperature if present, in (n1, n2, n3) order."),
    ("return dict(geometry=geom", "The same dictionary as `read_pluto_binary`, so the rest of `load_pluto` is shared by both readers."),
]),
("def _find_var", "_find_var()", [
    "Case-insensitive lookup of a variable name.",
]),
("def load_pluto", "load_pluto(): read and find the units", [
    (("if os.path.isdir(path)", "base = os.path.dirname"), "A directory is binary output; a file is VTK."),
    (("if reader == \"pypluto\"", "using the built-in reader"), "With `reader='pypluto'` the data are read by pyPLUTO. If that fails (missing module, unsupported layout, a bug in an old copy), the built-in reader is used and a message says so."),
    (("units = dict(PLUTO_DEFAULT_UNITS)", "break"), "Start from PLUTO's defaults and look for definitions.h next to the data or one level up."),
    (("if definitions is not None", "src = os.path.relpath"), "Take the units from definitions.h."),
    (("for k, v in ((\"UNIT_DENSITY\"", "src = \"user arguments\""), "Explicit arguments override everything."),
]),
("v = d[\"vars\"]", "load_pluto(): physical fields", [
    ("rho_cgs = rho * UD", "Density in g cm$^{-3}$."),
    ("ne = rho_cgs * 1e3", "$n_e=\\rho/(\\mu_e m_p)$ in m$^{-3}$ (g cm$^{-3}$ $\\times10^3$ = kg m$^{-3}$)."),
    (("if te_const is not None", "tsrc = \"T/Te variable"), "A constant temperature, or a T/Te variable from the file."),
    (("p_cgs = prs * UD * UV**2", "te = p_cgs"), "Otherwise $T=p\\,\\mu m_p/(\\rho k_B)$ in cgs ($m_p$ in g, $k_B$ in erg/K)."),
]),
("Lu = (r_star", "load_pluto(): grid", [
    ("Lu = (r_star", "Output length unit in metres: $R_\\odot$, or $R_\\ast$ when `r_star` is given (stellar runs)."),
    ("L = UL / (Lu * 100.0)", "Code length unit expressed in that unit ($U_L$ in cm divided by $R_\\odot$ or $R_\\ast$ in cm)."),
    (("if geom == \"SPHERICAL\"", "r_body=r_body, dtype=dtype, length_unit=Lu)"), "Spherical: radii in $R_\\odot$ (or $R_\\ast$), angles unchanged; one $\\phi$ cell means axisymmetric. The grid records its length unit."),
    (("elif geom == \"CARTESIAN\"", "r_body=1.0 if r_body"), "Cartesian: all axes in $R_\\odot$ (or $R_\\ast$)."),
]),
("if verbose:", "load_pluto(): report", [
    (("print(f\"radiosynth: PLUTO", "print(f\"            {g}\")"), "Print the units used, the plasma model and the grid."),
    (("if geom == \"SPHERICAL\" and not", "pass definitions="), "Warn if the inner radius is not near 1 $R_\\odot$, the usual sign of a wrong length unit."),
    (("rr = g.r", "(median over angles)"), "Typical $n_e$, $T_e$ at $r\\approx1.1\\,R_\\odot$ as a sanity check."),
]),
]))

# ============================================================================
MODULES.append(("__main__.py", "__main__.py --- command-line interface",
r"""Makes the package runnable as `python -m radiosynth INPUT --freq ...`. It parses the
options, loads the model with `load_mhd`, calls `make_radio_image`, and writes the
requested files.""",
[
('"""Command-line', "Module docstring and imports", []),
("def main(argv=None)", "Options: image and physics", [
    ("ap.add_argument(\"input\"", "Model file or PLUTO directory."),
    (("--freq", "--up"), "Frequencies, observer (or `--date`, `--frame`, `--phi-offset` to place it at the Earth for a given time), image size, pixels and up vector."),
    (("--emission", "--kappa-over-mu"), "Emission model, Gaunt factor, transfer mode and refraction switches."),
    (("--r-integration", "--workers"), "Geometry and numerical controls, and the number of processes."),
    (("st = ap.add_argument_group", "--phase"), "Stellar options: stellar radius (coordinates in $R_\\ast$), distance in parsec (parallel rays, flux in mJy), inclination and rotational phase of the view."),
    (("--density-var", "--save-grid"), "Input interpretation for BATS-R-US and generic files, and resampling."),
]),
("pl = ap.add_argument_group", "Options: PLUTO and output", [
    (("--nout", "--mmw"), "PLUTO output number, format, definitions file, units and plasma parameters."),
    (("--out", "--plot"), "Output prefix, file formats and plots."),
]),
("a = ap.parse_args", "Running", [
    (("rs = None", "box = ((b[0]"), "Convert the resampling options to tuples."),
    ("grid = load_mhd(", "Load the model; every reader-relevant option is passed and `load_mhd` keeps the ones that apply."),
    ("grid.to_npz", "Optionally cache the grid."),
]),
("gaunt = a.gaunt", "Imaging and output", [
    ("gaunt =", "'dulk' or a number."),
    (("if a.date:", "observer = observer_for_date"), "With `--date`, the observer is computed by `observer_for_date`."),
    (("if a.inclination is not None", "observer = stellar_observer"), "With `--inclination`, the view direction comes from the inclination and phase."),
    (("if a.distance_pc:", "mJy"), "For a star, print the flux density at each frequency in mJy."),
    ("img = make_radio_image(", "Compute the image."),
    (("if \"npz\" in a.formats", "img.plot(\"intensity\""), "Write the requested files and plots."),
]),
]))

# ============================================================================
MODULES.append(("__init__.py", "__init__.py --- package interface",
r"""Defines what `import radiosynth as rs` provides.""",
[
('"""radiosynth', "Public names", [
    ("from .constants import", "Constants and plasma-frequency helpers."),
    ("from .grid import", "The two grid classes."),
    ("from .raytrace import", "The tracer and the status names."),
    ("from .image import", "Image driver, result class, path tracing, geometry and frequency parsing."),
    ("from .readers import", "Generic and BATS-R-US readers."),
    ("from .pluto import", "PLUTO readers (built-in, VTK and pyPLUTO)."),
    ("from . import figures", "The figure module, available as `rs.figures`."),
]),
]))

# ============================================================================
MODULES.insert(len(MODULES) - 2, ("figures.py", "figures.py --- figures in the style of Moschou et al. (2018)",
r"""Plotting routines for publication figures: a synthetic map compared with an
observation (Fig. 2 of the paper), a single-frequency map (Fig. 3) and a time
sequence of CME images at several frequencies (Fig. 5), plus an animation writer.
Coordinates can be shown in $R_\odot$ or helioprojective arcsec.""",
[
('"""Publication-style figures', "Module docstring and imports", [
    "Describes the four figure functions and the coordinate conventions.",
    ("ARCSEC_PER_RAD", "Radians to arcseconds."),
]),
("def earth_observer", "earth_observer()", [
    "Places the observer at a given heliographic latitude B0 and longitude L0. For a model in a heliographic frame with $z$ along the solar rotation axis, the image then has solar North up and West to the right, the same as observed maps.",
    ("return (distance_rsun", "$\\vect O=D(\\cos B_0\\cos L_0,\\ \\cos B_0\\sin L_0,\\ \\sin B_0)$."),
]),
("def _julian_day", "_julian_day()", [
    (("if isinstance(time, (int, float", "return float(time)"), "A number is taken as a Julian day."),
    (("if isinstance(time, str)", "time = time.astimezone"), "An ISO string is parsed; a time-zone-aware datetime is converted to UTC."),
    ("return 2451545.0", "Julian day = 2451545.0 (J2000.0, 2000 Jan 1 12:00) plus elapsed days. UTC is used in place of TT; the 69 s difference changes $L_0$ by less than $0.01^\\circ$."),
]),
("def sun_ephemeris", "sun_ephemeris(): B0, L0, P and distance", [
    (("if use_sunpy:", "pass"), "Use sunpy if it is installed."),
    ("T = (jd - 2451545.0) / 36525.0", "Julian centuries since J2000.0."),
    (("Ls = 280.46646", "true_long = Ls + C"), "Sun's geometric longitude (Meeus ch. 25): mean longitude $L$, mean anomaly $M$, equation of centre $C$."),
    (("omega = 125.04", "lam = true_long"), "Apparent longitude $\\lambda$, corrected for aberration and the main nutation term."),
    (("e = 0.016708634", "R_au ="), "Eccentricity and Sun--Earth distance in AU."),
    ("eps = (23.0", "True obliquity of the ecliptic."),
    (("theta = (jd - 2398220.0)", "K = 73.6667"), "Meeus ch. 29: rotation angle $\\theta$ (Carrington rate, 25.38 d sidereal), inclination $I=7.25^\\circ$ of the solar equator, and longitude $K$ of its ascending node."),
    (("x = math.degrees", "P = x + y"), "Position angle $P$ of the solar rotation axis, measured from celestial north. You need it to rotate RA/Dec images to solar-north-up."),
    ("B0 = math.degrees", "$B_0=\\arcsin(\\sin(\\lambda-K)\\sin I)$, the heliographic latitude of the disk centre."),
    (("eta = math.degrees", "L0 = (eta - theta) % 360.0"), "$L_0=\\eta-\\theta$, the Carrington longitude of the disk centre (the sub-Earth point)."),
    ("return dict(B0=B0", "Distance converted to $R_\\odot$. Checked against Meeus' worked example 29.a to $0.02^\\circ$."),
]),
("def observer_for_date", "observer_for_date()", [
    "Places the observer at the Earth's position for a date, in the model frame.",
    ("lon = (eph[\"L0\"]", "Carrington frame: longitude $L_0+$offset. Stonyhurst frame (model longitude 0 faces the Earth): just the offset."),
    ("return earth_observer", "Observer at latitude $B_0$, that longitude, and the true distance (or the given one)."),
]),
("def rsun_to_arcsec", "rsun_to_arcsec(), image_extent(), limb_radius()", [
    ("return np.arctan", "Helioprojective angle $T_x=\\arctan(X/D)$ of an image-plane coordinate."),
    (("def image_extent", "return list(e)"), "Pixel-edge extent of a RadioImage, for `imshow`, in $R_\\odot$ or arcsec."),
    (("def limb_radius", "math.asin"), "Apparent radius of the photosphere: 1 $R_\\odot$, or $\\arcsin(1/D)$ in arcsec (about 960'' at 1 AU)."),
]),
("def _freq_index", "Helper functions", [
    ("def _freq_index", "Select a frequency by index, by value in Hz, or by a string such as '125MHz' (nearest frequency in the image)."),
    ("def _data", "Intensity or brightness temperature for one frequency."),
    ("def smooth_to_beam", "Convolve with a circular Gaussian beam: $\\sigma=\\text{FWHM}/(2\\sqrt{2\\ln2})$ in pixels. Use this to compare with an instrument of finite resolution."),
    ("def log_levels", "`n` logarithmically spaced contour levels below the maximum, over a chosen number of decades."),
    ("def decade_levels", "One contour per power of ten, as in Fig. 2 of the paper."),
]),
("def read_fits_map", "read_fits_map(): observed FITS maps", [
    ("from .fitsio import read_fits", "Uses astropy when installed, otherwise the small built-in reader."),
    (("def ax(i, n)", "return (np.array"), "Extent of each axis from the linear WCS keywords: $(\\text{pixel edge}-\\text{CRPIX})\\,\\text{CDELT}+\\text{CRVAL}$, converted to arcsec using CUNIT."),
    (("rot = float", "data = rotate"), "If the map is rotated (CROTA2), rotate it so that solar North is up."),
]),
("def figure_single_map", "figure_single_map(): Fig. 3 style", [
    (("k = _freq_index", "d = smooth_to_beam"), "Select the frequency and optionally smooth to a beam."),
    (("if log and pos.size", "im = ax.imshow(d, origin"), "Linear colour scale by default (as in the paper), logarithmic if `log=True`."),
    (("lev = log_levels", "ax.contour"), "White log-spaced contours."),
    (("R = limb_radius", "ax.plot(R * np.cos(t)"), "Limb circle."),
    ("ax.text(0.03, 0.97", "The 'f = ... MHz' label in the upper-left corner."),
]),
("def figure_compare_observation", "figure_compare_observation(): Fig. 2 style", [
    "Left: the synthetic map on a log scale with its contours. Right: the observation with the same synthetic contours and the limb overplotted.",
    (("syn = _data", "syn_s = smooth_to_beam"), "Synthetic map in arcsec, smoothed to the instrument beam if requested."),
    (("if isinstance(obs, str)", "odata, oext = np.asarray"), "The observation is a FITS file or an array with its extent."),
    (("if isinstance(levels, str)", "lev = np.asarray"), "Contour levels: one per decade, log-spaced, or given explicitly."),
    (("im0 = axs[0].imshow", "cb.ax.axhline"), "Synthetic panel; the contour levels are also marked on its colour bar, as in the paper."),
    (("im1 = axs[1].imshow", "axs[1].plot(R"), "Observation panel with the synthetic contours and a green limb."),
    (("lim = fov_arcsec", "a.set_ylabel"), "Common field of view and axis labels."),
]),
("def make_time_series", "make_time_series()", [
    (("for i, s in enumerate(snapshots)", "out.append(img)"), "For each snapshot (file, PLUTO directory, or grid): load it, make the image with the same settings and optionally save it. The result is a list of RadioImages ordered in time."),
]),
("def figure_time_series", "figure_time_series(): Fig. 5 style", [
    ("ks =", "Frequencies to show (columns)."),
    (("alld = np.concatenate", "lev = np.geomspace"), "One logarithmic colour scale and one set of contour levels for all panels, so the panels can be compared directly."),
    (("for r, im in enumerate(images)", "a.set_yticks([])"), "Rows are times and columns frequencies: image, contours, frequency title, time stamp in the first column."),
    (("cb = fig.colorbar", "cb.ax.axhline"), "Shared colour bar with the contour levels marked."),
]),
("def animate_time_series", "animate_time_series()", [
    (("def draw(i)", "return []"), "Draw one frame: image, contours, frequency and time."),
    (("ani = animation.FuncAnimation", "ani.save"), "Write a GIF with Pillow, or an MP4 with ffmpeg if the file name ends in .mp4."),
]),
("def _regrid_to_obs", "_regrid_to_obs()", [
    (("xs = rsun_to_arcsec", "d = smooth_to_beam"), "Synthetic pixel axes in arcsec; smooth to the beam."),
    (("f = RegularGridInterpolator", "return f("), "Bilinear interpolation onto the observed pixel centres; NaN outside the synthetic field of view."),
]),
("def fit_longitude_offset", "fit_longitude_offset()", [
    (("if offsets is None", "odata, oext = np.asarray"), "Trial offsets (default $-60^\\circ$ to $60^\\circ$ in $5^\\circ$ steps) and the observed map."),
    (("tx = oext[0]", "mask ="), "Observed pixel centres and the comparison region $r<$ `r_mask` apparent solar radii."),
    (("for i, off in enumerate(offsets)", "syn = _regrid_to_obs"), "For each offset: move the observer in longitude, make the synthetic map, and put it on the observed pixels."),
    (("if use_log:", "corr[i] = np.corrcoef"), "Pearson correlation of the logarithms. It is insensitive to the absolute calibration of the observation and gives faint and bright structures comparable weight."),
    (("ib = int(np.nanargmax(corr))", "best_ref = float(np.clip"), "Best grid offset, refined by the vertex of a parabola through the best point and its two neighbours."),
    (("if plot_path is not None", "plt.close(fig)"), "Optional plot of correlation against offset."),
]),
("def compare_with_observation", "compare_with_observation()", [
    (("k = _freq_index", "syn = _regrid_to_obs"), "Put the beam-smoothed synthetic map on the observed pixels."),
    (("TY, TX = np.meshgrid", "r = np.hypot"), "Distance from the disk centre in apparent solar radii."),
    (("disk =", "inner ="), "Regions: disk ($r<1$), limb ($1\\le r<1.3$), and both, for the correlation."),
    (("def prof(m)", "else np.nan for a, b"), "Azimuthally averaged radial profile."),
    (("def r_half(p)", "return float(np.interp(half"), "Size of the radio Sun: the outermost radius where the profile is still at least half its maximum. This is robust against limb brightening and central dips."),
    ("pa = np.degrees", "Position angle of each pixel, measured from solar north counter-clockwise (towards East, i.e. $-X$), as is usual for solar images."),
    (("def prof_sector", "sel = ok & (dpa <= sector_halfwidth)"), "Radial profile restricted to a sector of $\\pm$ `sector_halfwidth` around a position angle; the angle difference is wrapped to $[-180^\\circ,180^\\circ]$."),
    (("for tag, m in ((\"obs\"", "res[f\"elongation_{tag}\"]"), "Profiles and radio-Sun size in the N, E, S, W sectors, and the elongation (equatorial/polar size). This is the main measure for axisymmetric (2.5D) models."),
    (("res = dict(syn=syn", "res[\"ratio_disk\"]"), "Collect the maps, the mean disk and limb values, the log correlation, the profiles and the sizes."),
    (("if plot_path is not None", "plt.close(fig)"), "Optional figure: both radial profiles and the residual map (synthetic minus observed)."),
]),
]))

MODULES.insert(len(MODULES) - 2, ("fitsio.py", "fitsio.py --- FITS input/output without astropy",
r"""Small FITS reader and writer for single images. astropy is used when it is
installed; otherwise these routines handle the simple primary-HDU images used for
solar radio maps and for radiosynth's own output.""",
[
('"""Minimal FITS', "Constants", [
    ("_BLOCK = 2880", "FITS files are written in blocks of 2880 bytes."),
    ("_DT =", "BITPIX code to NumPy data type (FITS is big-endian)."),
]),
("def _parse_value", "_parse_value()", [
    "Turns the value field of an 80-character header card into a string, a boolean, an integer or a float, dropping the comment after '/'.",
]),
("def read_fits", "read_fits()", [
    (("try:", "pass"), "Use astropy if available."),
    (("while True:", "break"), "Read 2880-byte header blocks until the END card, collecting the KEY = value cards."),
    (("naxis = hdr.get", "nbytes ="), "Image dimensions and size."),
    (("data = np.frombuffer", "return data, hdr"), "Read the data (FITS order: NAXIS1 fastest, giving a (ny, nx) array) and apply BSCALE/BZERO."),
    ("pos += -(-nbytes", "Skip to the next HDU if a later one was requested."),
]),
("def write_fits", "write_fits()", [
    (("def card(k, v)", 'return f"{k:<8}= {s:<20}".ljust(80)'), "Format one 80-character card for booleans, integers, floats and strings."),
    (("cards = [", 'cards.append("END"'), "Mandatory keywords (SIMPLE, BITPIX=-32, NAXIS, NAXISn), the user's keywords, END."),
    (("h += ", "fh.write(body)"), "Pad the header and the big-endian float32 data to whole 2880-byte blocks."),
]),
]))

# ============================================================================
MODULES.insert(len(MODULES) - 2, ("stellar.py", "stellar.py --- stellar radio emission",
r"""Tools for other stars: viewing geometry from inclination and rotational phase,
images of a star at a distance in parsec (parallel rays), spectra and spectral
indices, rotational modulation, and the analytic optically thick wind solution
used for validation. Model coordinates are in stellar radii; the grid's
`length_unit` holds $R_\ast$ in metres.""",
[
('"""Stellar radio emission', "Module docstring and constants", [
    "Explains the conventions: rotation axis = model $z$; inclination $i$ between the axis and the line of sight; phase $\\phi$ = model longitude facing the observer.",
    ("PC = ", "One parsec in metres."),
]),
("def stellar_observer", "stellar_observer()", [
    ("return (distance * math.sin(i)", "Direction $\\hat n=(\\sin i\\cos\\phi,\\ \\sin i\\sin\\phi,\\ \\cos i)$ times a distance. With `distance_pc` the rays are parallel and only the direction matters."),
]),
("def _default_fov", "_default_fov()", [
    "Default field of view: the largest square inside the model domain.",
]),
("def make_stellar_image", "make_stellar_image()", [
    (("fov = _default_fov", "kw.setdefault(\"r_integration\""), "Defaults for stars: full radiative transfer (stellar coronae and winds are often optically thick) and an integration sphere covering the whole model."),
    ("return make_radio_image", "The image itself: parallel rays from the direction given by $i$ and $\\phi$, flux at `distance_pc`."),
]),
("def spectral_index", "spectral_index()", [
    ("alpha = np.diff", "$\\alpha=\\Delta\\ln S/\\Delta\\ln\\nu$ between neighbouring frequencies, at their geometric mean."),
]),
("def stellar_spectrum", "stellar_spectrum()", [
    (("img = make_stellar_image", "nu_mid, alpha = spectral_index"), "One multi-frequency image gives the spectrum (mJy) and the local spectral index."),
    (("if plot_path is not None", "plt.close(fig)"), "Optional figure: spectrum and spectral index, with reference values 2 (optically thick), 0.6 (ionised wind) and $-0.1$ (optically thin free--free)."),
]),
("def rotational_modulation", "rotational_modulation()", [
    (("for ph in phases", "flux.append"), "One image per rotational phase; the flux densities form the light curve."),
    (("if plot_path is not None", "plt.close(fig)"), "Optional light-curve figure."),
]),
("def wind_optical_depth", "wind_optical_depth(): analytic reference", [
    ("k1 = EmissionModel", "$\\kappa_1=\\kappa/n_e^2$ for the given $T_e$, frequency and Gaunt factor."),
    ("tau = A * math.pi / (2.0 * p**3)", "Along an infinite straight line at impact parameter $p$: $\\tau=\\kappa_1 n_0^2r_0^4\\int dl/(p^2+l^2)^2=\\kappa_1n_0^2r_0^4\\,\\pi/(2p^3)$."),
    (("if r_body_m is not None", "tau[inside] = A *"), "Rays that hit the star see only the wind in front of it. The integral then runs from $l_0=\\sqrt{r_\\ast^2-p^2}$, using the antiderivative $l/(2p^2(p^2+l^2))+\\arctan(l/p)/(2p^3)$."),
]),
("def wind_flux_density", "wind_flux_density(): analytic flux of an optically thick wind", [
    "With $\\tau(p)=a/p^3$ the flux is $S=2\\pi B\\int_0^\\infty(1-e^{-a/p^3})p\\,dp/D^2=\\pi\\Gamma(1/3)\\,B\\,a^{2/3}/D^2$. Since $a\\propto\\nu^{-2}$ and $B\\propto\\nu^2$, this gives $S\\propto\\nu^{2/3}$, the classic wind spectrum (Wright \\& Barlow 1975; Panagia \\& Felli 1975). It becomes $\\nu^{0.6}$ when the Gaunt factor's weak frequency dependence is included.",
    ("return math.pi * math.gamma", "The closed form; checked against direct quadrature to $10^{-7}$."),
]),
]))
