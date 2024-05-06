# Multi-objective Optimization for Selection of Clusterings Across Time
Method for clustering of time series over time based on dual-objective optimization.

## Introduction
This repository contains the implementation of MOSCAT (Multi-objective Optimization for Selection of Clusterings Across Time).

## Datasets
For our experiments we used following publicly accessable datasets:
- ERing: https://github.com/time-series-machine-learning/tsml-java/tree/master/src/main/java/experiments/data/mtsc/ERing
- Motions: https://github.com/time-series-machine-learning/tsml-java/tree/master/src/main/java/experiments/data/mtsc/BasicMotions
## Getting Started
```
virtualenv venv
source venv/bin/activate
pip install -e .
```
## Experiments
The experiments are in the folder "experiments". To re-run an experiment, go into corresponding folder and do following:
```
python do_baseline_clusterings.py
python do_evol_clusterings.py
python do_moscat_clusterings.py
python calculate_stats.py
```
After re-running all clustering algorithms and recalculating stats, go to evaluation.ipynb in order to see the results.