# Overview figures (Figures 1 and 3 of RESULTS.md)

The two figures are rendered from stored data and need only numpy, scipy and
matplotlib:

```bash
python scripts/overview_figure/fig_study04_split.py        # -> figures/fig6_model_responses.*, fig7_posterior_misspecification.*
```

`fig_study04.py` holds the panel functions; run directly it writes the original
one-row layout (`figures/overview_1x4.*`), which `RESULTS.md` does not use.

The stored data, `results/overview_figure/fig_study04_data.npz` (every array
the panels plot) and `net_seed2.npz` (the network geometry of panel A), are
rebuilt from the study's stored outputs by the following pipeline, run from
the study root with `vmconf` and `porovasc` (study 01) installed. The argument
is the network seed; the stored data use seed 2, and panel E pools seeds 1-3
from `results/inference_d30_seed{1,2,3}.json`.

```bash
python scripts/overview_figure/build_fwd.py 2   # network and forward model (about 2.5 min) -> fwd_seed2.pkl
python scripts/overview_figure/post_grid.py 2   # grid posteriors of one representative repetition -> post_seed2.npz
python scripts/overview_figure/post_mean.py 2   # posteriors averaged over the 40 repetitions, added to post_seed2.npz
python scripts/overview_figure/make_data.py 2   # collects everything the figure plots -> fig_study04_data.npz
python scripts/overview_figure/save_net.py 2    # network geometry for panel A -> net_seed2.npz
```

Intermediates (`fwd_seed*.pkl`, `post_seed*.npz`) are written to
`results/overview_figure/` and are not stored. Panel A draws segments of
radius 40 µm and above in the gland and, in the enlargement, segments of 60 µm
and above at true radius. Panel B refits $(\mu,\eta)$ at $s=0.73$ and $0.83$,
the SWE-only 90% posterior bounds of $s$ averaged over the 40 repetitions,
by least squares to the generating phase velocity and attenuation at the 24
study frequencies; the refitted curves deviate from the generating ones by at
most 0.7% in phase velocity and 2.9% in attenuation. Panel D averages the $(\mu, s)$ marginal posterior over the 40
noise realizations and draws 50% and 90% highest-density regions after light
smoothing (Gaussian filter of 0.8 grid cells, cubic upsampling by 4) for
display only. The scripts request Arial Narrow and fall back to the default
sans-serif font.
