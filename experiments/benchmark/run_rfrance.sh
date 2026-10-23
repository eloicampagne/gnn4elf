#!/bin/bash
# French benchmark under the WEAVE-UK protocol: three GNN lanes and one baseline lane.
cd "$(dirname "$0")"
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
lane() { for c in "$@"; do $PY -u benchmark_rfrance.py hpo --conv $c --trials 40 && $PY -u benchmark_rfrance.py train --conv $c --seeds 3; done; }
lane GATv2Conv TransformerConv GCNConv > results_rfrance_v2/lane1.log 2>&1 &
lane GATConv GraphSAGE TAGConv > results_rfrance_v2/lane2.log 2>&1 &
lane ChebConv APPNP > results_rfrance_v2/lane3.log 2>&1 &
$PY -u benchmark_rfrance.py tuned_baselines --trials 40 > results_rfrance_v2/baselines.log 2>&1 &
wait
$PY -u benchmark_rfrance.py aggregate > results_rfrance_v2/aggregate.log 2>&1
echo done > results_rfrance_v2/ALL_DONE
