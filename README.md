# gnn4elf

Code for the paper *A Diagnostic Protocol for GNNs in Electricity Load Forecasting: Separating Parameter Sharing from Neighborhood Access* (Campagne, Amara-Ouali, Goude, Zehavi, and Kalogeratos), [arXiv:2610.35074](https://arxiv.org/abs/2610.35074).

A GNN that forecasts a collection of load series shares parameters across series and reads neighboring series. The protocol of the paper separates the two mechanisms by comparing the GNN with a per-node network, a network shared across nodes without neighbors, and the same GNN without edges, and it tests neighbor identity by degree-preserving rewiring. The models, graph constructions and training loops come from [GraphToolbox](https://github.com/eloicampagne/graphtoolbox) (Campagne et al., ICTAI 2026, [arXiv:2609.24609](https://arxiv.org/abs/2609.24609)); this repository holds the experiment scripts, their configurations and the summaries behind the tables of the paper.

## Installation

```sh
git clone https://github.com/eloicampagne/gnn4elf.git
cd gnn4elf
pip install -r requirements.txt
```

The learned-adjacency control (`experiments/decomposition/learned_adjacency.py`) also needs the MTGNN code:

```sh
git clone https://github.com/nnzhan/MTGNN external/MTGNN
```

The foundation-model baselines `t0_baseline.py` and `timestral_baseline.py` need `tfc-t0` (0.6.0) and `timestral` (1.0.0), which are not in `requirements.txt`; the runs of the paper used a separate environment for them.

## Data

The processed REGIONS-FR and WEAVE-UK datasets and the synthetic benchmarks are on Zenodo, [doi:10.5281/zenodo.21977576](https://doi.org/10.5281/zenodo.21977576). The ELECTRICITY collection is the one distributed with Informer (Zhou et al., 2021). [`data/README.md`](data/README.md) gives the expected file layout.

## Layout

```
configs/                    hyperparameters selected by the benchmark searches
experiments/
  _paths.py                 repository paths; puts every experiment folder on sys.path
  data_prep/                caches of the per-node tasks (data/cache/)
  benchmark/                tuned architecture benchmark and covariate-aware baselines
  french_regime/            four node-wise models on REGIONS-FR as the training set contracts
  synthetic/                controlled denoising study
  decomposition/            sharing/neighborhood decomposition on WEAVE-UK and ELECTRICITY, and its controls
  aggregation/              experts for the online aggregation
  interpretability/         latent representations and explanation graphs
figures/                    scripts of the figures of the paper (written to figures/out/)
graph_representations_*/    adjacency matrices of the candidate graphs
results_*/                  JSON summaries read by the tables; raw forecasts and checkpoints are not versioned
legacy/                     pipeline of arXiv:2507.03690v3 (28 WEAVE substations)
```

Scripts resolve every path from the repository root, so they can be launched from any directory, for instance `python experiments/decomposition/eval_metrics.py`.

## Correspondence with the paper

| Paper | Scripts |
|---|---|
| Section 6.1, Tables 2 and 3, Table H.1 (hyperparameters) | `benchmark/benchmark_rfrance.py`, `benchmark/benchmark_weave500.py` (driven by `run_rfrance.sh`, `run_bench500.sh`), `benchmark/extra_baselines.py` (ARIMA-X, Chronos-2), `benchmark/t0_baseline.py`, `benchmark/timestral_baseline.py` |
| Section 6.2, Figure 2, Tables C.1 and C.2, Figure C.3 | `french_regime/ib_lib.py`, `french_regime/expe_inductive_bias.ipynb`, `french_regime/french_nrmse.py` |
| Section 6.3, Figure 3 | `synthetic/synth_gnn.py`, `synthetic/synth_sweep.py` |
| Section 6.4, Table 4, Figure 4 | `decomposition/eval_metrics.py`, `decomposition/weave_nodes_metrics.py`, `decomposition/ltsf_nodes.py`, `decomposition/increment_intervals.py` |
| Appendix E, Table E.1 (node count) | `decomposition/weave_nodes_metrics.py`, `decomposition/ltsf_nodes.py` |
| Appendix F, Tables F.1 to F.4 (controls) | `rewire_test.py` and `extra_controls.py` (rewiring, graph estimated inside the window), `local_references.py` and `seed_node_intervals.py` (ridge and persistence references), `review_controls.py` (MLP square, other periods), `width_replication.py`, `homophily.py`, `unshared_graph.py`, `recurrent_baseline.py` (GRU), `learned_adjacency.py` (MTGNN), all in `decomposition/` |
| Section 6.5, Appendix B | `aggregation/aggreg_experts.py`, `aggregation/add_expert.py`, `aggregation/gam_expert.py`, `aggregation/try_convs.py` |
| Appendix G (representations) | `interpretability/extract_weave_latents.py`, `interpretability/interpretability.ipynb`, `interpretability/expe_interpretability.ipynb`; the REGIONS-FR latents come from GraphToolbox, `examples/load/compare_latent_representations.py` |

Table and figure numbers follow the preprint.

## Citation

```bibtex
@article{campagne2026diagnostic,
  author  = {Campagne, Eloi and Amara-Ouali, Yvenn and Goude, Yannig and Zehavi, Itai and Kalogeratos, Argyris},
  title   = {A Diagnostic Protocol for {GNNs} in Electricity Load Forecasting: Separating Parameter Sharing from Neighborhood Access},
  journal = {arXiv preprint arXiv:2610.35074},
  year    = {2026}
}

@inproceedings{campagne2026graphtoolbox,
  author    = {Campagne, Eloi and Amara-Ouali, Yvenn and Goude, Yannig and Kalogeratos, Argyris},
  title     = {{GraphToolbox}: A Configurable {Python} Framework for Graph Neural Network Forecasting},
  booktitle = {IEEE International Conference on Tools with Artificial Intelligence (ICTAI)},
  year      = {2026}
}
```

## License

GPL-3.0, see [LICENSE](LICENSE).
