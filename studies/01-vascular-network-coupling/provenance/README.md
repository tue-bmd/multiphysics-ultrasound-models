# Coupling-run source

The records in `runs/coupling/` were generated with source hash
`5fc58328873ae08f7eaf86d37c5897c82cad9c9ed30bcae564f73eaa708467c8`.
The corresponding source snapshot is stored as
`generating-source-5fc58328.tar.gz`.

A representative record can be recomputed with:

```bash
python -m porovasc.reproduce runs/coupling --seed 100 --rve 0 --arm baseline
```

The same command accepts `--arm tort_1.45` for the tortuosity perturbation.
