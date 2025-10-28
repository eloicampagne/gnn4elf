# Data directory — Instructions

This README explains how to place the data files expected by the project.

This project accompanies the paper "Graph Neural Networks for Electricity Load Forecasting" and is intended to reproduce the data setup used in that work. It also integrates with the GraphToolbox package developed by the project author, which provides the graph utilities and modeling components used in the experiments. Please cite the paper and reference the GraphToolbox repository when reusing these datasets or scripts.

## Location
Place your data files in:
`gnn4elf/data`

You can also reference this folder by its relative path from the project root: `./data/`.

## Accepted formats
Common supported formats: `.csv` and `.parquet`.

## Recommended structure
- data/
    - weave/        -> UK dataset
        - train_weave.csv   -> training data (split)
        - test_weave.csv    -> test data (split)
    - rfrance/      -> French regions dataset
        - train2.csv   -> training data (split)
        - test2.csv    -> test data (split)
    - README.md      -> this file

## Best practices
- Do not commit large data files to Git: add them to `.gitignore`.
- If files are large, use external storage (S3, Drive, LFS).
- Ensure read permissions are set: `chmod 644 data/*` if needed.
