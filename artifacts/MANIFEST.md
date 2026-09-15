# Stored outputs

The repository includes the outputs used in the three result summaries.

| Study | Output | Contents | Main command |
|---|---|---|---|
| 01 | `runs/coupling/` | 90 records from five prostate-gland realizations, two sampling volumes, and nine conditions | `python -m porovasc.run_coupling --out runs/coupling --n-real 5 --n-rve 2 --n-inputs 12 --t-end 163.84` |
| 02 | `results/` | identifiability matrix, profiles, discrepancy analyses, and posterior samples | `fieldid matrix`, `fieldid profiles`, `fieldid mcmc` |
| 02 | `results/truth_sweep_r5/` | 360 fits across nine parameter settings, eight configurations, and five noise realizations | `fieldid truth-sweep --out results/truth_sweep_r5 --n-rep 5` |
| 03 | `results/` | comparison arms, coverage analyses, and posterior samples | `operatorid matrix`, `operatorid coverage`, `operatorid mcmc` |

The effective configurations are stored with the outputs or in each study's
`configs/` directory. The corresponding figures and numerical summaries are
described in the study-level `RESULTS.md` files.
