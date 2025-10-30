# gnn4elf

Graph neural network code and forecasts accompanying the paper "Graph Neural Networks for Electricity Load Forecasting" (gnn4elf). This repository implements training and evaluation pipelines built on top of the [GraphToolbox package](https://github.com/eloicampagne/graphtoolbox).

## Highlights
- Implements models and training scripts used in the paper.
- Reproducible experiments and saved forecasts.
- Uses GraphToolbox for graph construction and utilities.

## Repository structure

```python
gnn4elf
├── configs/                         # experiment configuration files  
├── data/                            # datasets folder  
├── graph_representations_rfrance/   # graph representations for rfrance dataset  
├── graph_representations_weave/     # graph representations for weave dataset
├── results_chronos/                 # saved forecasts (chronos)  
├── results_rfrance/                 # saved forecasts for rfrance (gnn)  
├── results_tabpfn/                  # saved forecasts (tabpfn)  
├── results_tirex/                   # saved forecasts (tirex)  
├── results_weave/                   # saved forecasts for weave (gnn)  
├── scripts/                         # helper scripts
│   ├── optim_ff.py                  # optimization / hyperparam search (feed-forward variant)
│   ├── optim.py                     # optimization / hyperparam search
│   ├── model.py                     # model definitions (FF architecture)
│   ├── multi.py                     # multitask / multi-node helpers
│   ├── tirex_util.py                # TiREX utility functions
│   ├── tirex-zs-sequence-test.py    # TiREX training script
│   ├── train_gpu_ff.py              # GPU training script (feed-forward experiments)
│   └── train_gpu.py                 # GPU training script           
├── .gitignore  
├── expe.ipynb                       # exploratory notebook with experiments  
├── optim_ff.py                      # optimization / hyperparam search (feed-forward variant)  
├── optim.py                         # optimization / hyperparam search  
├── model.py                         # model definitions (FF architecture)  
├── multi.py                         # multitask / multi-node helpers  
├── README.md  
└── fm_expe.ipynb                    # exploratory notebook with foundation models experiments
````

## Requirements
- Python 3.9+
- PyTorch
- GraphToolbox (required; see [GraphToolbox docs](https://eloicampagne.fr/graphtoolbox) for installation)
- [TiREX](https://github.com/NX-AI/tirex) (for baseline)
- [Chronos](https://github.com/amazon-science/chronos-forecasting) (for baseline)
- Usual scientific stack: numpy, pandas, scikit-learn, matplotlib, seaborn
- (Optional) Jupyter for notebooks

## Quick start
1. Prepare data
    - Download the datasets from [Zenodo](https://zenodo.org/records/17453409?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJkMzEwODAzLWM2NDMtNDljZi05MTA1LTA3Mjk1NWQ5YmRhMCIsImRhdGEiOnt9LCJyYW5kb20iOiIzN2RjMjZiNDhiNzc5YzQ3NzQ5NTQ5ZDZmN2E0YTc3ZCJ9.WIfAN3nX9wRLfaAQim_BfQmiqiPkwKRKGr-lF2m2cOJM0d0O8rofN76WmcMCE6Pq-ushBaoIB2yZN_C3uKsxWQ).
    - Place dataset files under `data/` following the preprocessing expected by the scripts.
    - Put graph representations in the matching folders: `graph_representations_weave/` and `graph_representations_rfrance/`.

2. Hyperparameter search / optimization:
    ```bash
    python optim.py 
    python optim_ff.py
    ```

3. Configure an experiment
    - Edit or add a config file in `configs/` (the code reads configuration dictionaries; check existing examples).

4. Run training on GPU:
    ```bash
    # standard training
    python train_gpu.py --data_number $SLURM_ARRAY_TASK_ID --config configs/your_config.json

    # feed-forward variant
    python train_gpu_ff.py --data_number $SLURM_ARRAY_TASK_ID --config configs/your_ff_config.json
    ```
    Note that the python functions were designed to be run on HPC with SLURM.

5. Inspect results
    - Results and forecasts are saved in `results_rfrance/` or `results_weave/` depending on dataset.
    - Use `expe.ipynb` for exploratory analysis and plotting.

## Citation
If you use this code in your work, please cite the corresponding paper:

```bibtex
@misc{campagne2025graph,
    author = {Campagne, Eloi and Amara-Ouali, Yvenn and Goude, Yannig and Kalogeratos, Argyris},
    title = {Graph Neural Networks for Electricity Load Forecasting},
    year = {2025},
}
```

## Contact
- For questions, open an issue in the repository or contact the authors listed in the paper.

