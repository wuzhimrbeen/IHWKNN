# External baseline sources

The recent neural baselines remain under their original licences and should be obtained from the authors' repositories.

Place the repositories as follows before running the corresponding adapters:

```text
external/MDGCN-main/   official MDGCN repository: https://github.com/DHUDBlab/MDGCN
external/MCDR-main/    official MCDR repository: https://github.com/NiuDongjiang/MCDR
```

The AdaDR mechanism-preserving adaptation is self-contained in `experiments/new_baseline/ports/AdaDR/`. The common-protocol runners replace fold generation, candidate construction, and metric calculation while preserving the documented model computations. MDGCN uses released profiles on its four overlapping datasets and training-only profile selection on the other four. The eight-dataset MCDR comparison is explicitly the no-DDI adaptation because an equivalent external DDI matrix was not available for all datasets.

Do not relabel adapted runs as untouched reproductions of the source publications.
