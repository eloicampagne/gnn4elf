#!/bin/bash
# Hyperparameter search then final seeds for the eight convolutions, 500 feeders.
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
cd "$(dirname "$0")"
for c in GCNConv GraphSAGE GATConv GATv2Conv TransformerConv TAGConv ChebConv APPNP; do
  $PY benchmark_weave500.py hpo --conv $c --trials 40 2>&1 | grep -vE "it/s\]|it\]$|%\|" 
  $PY benchmark_weave500.py train --conv $c --seeds 3 2>&1 | grep -vE "it/s\]|it\]$|%\|"
done
echo DONE
