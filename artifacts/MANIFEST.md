# Stored outputs

The repository includes the outputs used in the four result summaries.

| Study | Output | Contents | Main command |
|---|---|---|---|
| 01 | `runs/coupling/` | 90 records from five prostate-gland realizations, two sampling volumes, and nine conditions | `python -m porovasc.run_coupling --out runs/coupling --n-real 5 --n-rve 2 --n-inputs 12 --t-end 163.84` |
| 02 | `results/` | identifiability matrix, profiles, discrepancy analyses, and posterior samples | `fieldid matrix`, `fieldid profiles`, `fieldid mcmc` |
| 02 | `results/truth_sweep_r5/` | 360 fits across nine parameter settings, eight configurations, and five noise realizations | `fieldid truth-sweep --out results/truth_sweep_r5 --n-rep 5` |
| 03 | `results/` | comparison arms, coverage analyses, and posterior samples | `operatorid matrix`, `operatorid coverage`, `operatorid mcmc` |
| 04 | `results/` | three CEUS response tables, three nominal inference files with 40 repetitions each, diagnostics, and the parameter-sweep summary | `python run_ceus.py`, `python run_inference.py`, `python run_diagnostics.py` |
| 04 | `results/true_value_sweep/` | 108 per-network inference files across 36 generating-parameter combinations; 81 files contribute to the reported 27-combination summary | `python run_true_value_sweep.py --mu-values 1500 2000 2500 --eta-values 0.5 1.0 1.5 --s-values 0.70 0.80 0.90 --reps 40` |

The effective configurations are stored with the outputs or in each study's
`configs/` directory. The corresponding figures and numerical summaries are
described in the study-level `RESULTS.md` files.
