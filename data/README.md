# Benchmark data

`archives/` contains the eight public DDA benchmark archives used by the
experiments plus the available drug/disease identifier archive. Run
`python prepare_data.py` from the repository root before executing experiments.

For every dataset, the model reads:

```text
data/<dataset>/ANMF/DiDrA.txt
data/<dataset>/ANMF/DrugSim.txt
data/<dataset>/ANMF/DiseaseSim.txt
```

The benchmark collection was obtained from the comparative evaluation
framework described in:

> Li et al. A comparative benchmarking and evaluation framework for
> heterogeneous network-based drug repositioning methods. *Briefings in
> Bioinformatics*. 2024. DOI: https://doi.org/10.1093/bib/bbae172

The five archives already present in the public repository retain the original
benchmark train/test files in addition to the three matrices. The revision
experiments do not use those legacy splits; all methods load the deterministic
folds produced by the shared evaluation protocol. The three scientific matrix
files in those archives were verified byte-for-byte against the matrices used
for the revision experiments.

`archives/SHA256SUMS.txt` records archive-level checksums. The source datasets
remain subject to the terms and licences of their original providers. Their
redistribution here is solely to make the exact benchmark inputs auditable;
users should cite the benchmark article and the original dataset publications.
