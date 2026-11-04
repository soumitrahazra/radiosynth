# radiosynth notes

Short, self-contained documents on the physics and the numerics of radiosynth, by mechanism.
Each folder has the PDF, its LaTeX source, the figures it needs, and the script that
computes those figures with radiosynth.

| Folder | Document | What it is |
|---|---|---|
| `bremsstrahlung/` | `radiosynth_bremsstrahlung.pdf` | Thermal free–free emission: every equation (coefficients, refraction, ray integration, thin and full transfer, 1D models, stellar wind) and the code that implements it |
| | `radiosynth_bremsstrahlung_story.pdf` | *The story of a radio image* — the same calculation told simply, step by step |
| `gyroresonance/` | `radiosynth_gyroresonance.pdf` | Thermal gyroresonance emission: magneto-ionic modes, derivation of the layer optical depth, magnetised free–free, polarised transfer, mode coupling, and the code that implements it |
| | `radiosynth_gyroresonance_story.pdf` | *The story of a sunspot at microwaves* — the same calculation told simply, with a model active region |
| | `radiosynth_gyroresonance_steps.pdf` | *Thermal gyroresonance radio emission in radiosynth* — step-by-step calculation from a 3D MHD simulation, with detailed derivations (Appleton–Hartree, layer optical depth) and one worked example through every step |
| `gyroresonance/simple/` | `gyroresonance_simple.pdf` | *Gyroresonance emission, step by step* — the calculation explained step by step together with `gyroresonance_simple.py`, a short standalone teaching code (numpy and matplotlib only) that reproduces `rs.make_gr_image` to 0.01 % for a flat box seen from above |

The full User Manual and the line-by-line Code Reference are in `doc/`.

## Rebuilding

```bash
cd doc/notes/gyroresonance            # or doc/notes/bremsstrahlung
python make_gr_figure.py              # figures (make_gr_story_figure.py, make_ff_figures.py)
pdflatex radiosynth_gyroresonance.tex # twice
```

The scripts use the radiosynth of this repository, so they can be run from anywhere.

## The simple gyroresonance code

```bash
cd doc/notes/gyroresonance/simple
python gyroresonance_simple.py        # the teaching code: prints T_b and fluxes, writes gyroresonance_simple_fig.png/.pdf
python check_with_radiosynth.py       # the same model with radiosynth: both give the same image
python make_side_view.py              # side_view.pdf (figure 1 of the note)
pdflatex gyroresonance_simple.tex     # twice; the program is printed in the appendix
```

`gyroresonance_simple.py` is a teaching version (a flat box seen from above, analytic model).
For images of real MHD models and any observer, use `rs.make_gr_image`. The test
`tests/test_simple_gyroresonance.py` checks that the two codes agree.
