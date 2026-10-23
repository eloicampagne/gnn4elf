# Data

`data/` is not versioned. The scripts expect the following files, all paths relative to the repository root.

```
data/
  rfrance/train2.csv, test2.csv              REGIONS-FR, 12 regions, 2015-2019 (Zenodo)
  weave/train_weave_500.csv, test_weave_500.csv
                                             WEAVE-UK, 500 feeders, 14-28 February 2024 (Zenodo)
  electricity/electricity.csv                ELECTRICITY, 321 clients, hourly (Informer distribution)
  cache/rfrance_tasks.npz, weave500_tasks.npz
                                             per-node tasks, built by experiments/data_prep/
```

REGIONS-FR and WEAVE-UK are deposited on Zenodo, [doi:10.5281/zenodo.21977576](https://doi.org/10.5281/zenodo.21977576). The caches are built from them:

```sh
python experiments/data_prep/build_cache.py         # data/cache/rfrance_tasks.npz
python experiments/data_prep/build_weave_cache.py   # data/cache/weave500_tasks.npz
```

`legacy/` reads the earlier 28-substation files `weave/train_weave_28.csv` and `weave/test_weave_28.csv`; its notebooks resolve `./data/` from the repository root.
