Experiment 1:
- Dataset: Random generated 2d Data
- Clustering: DBSCAN
- Snapshot quality: DBCV

Experiment 2:
- Dataset: EIKON
- Clustering: DBSCAN
- Snapshot quality: DBCV

Experiment 3:
- Dataset: Random 2d
- Clustering: Kmeans


Experiment 4:
- Multiple datasets
- KMeans
- Comparison of evolutionary clustering with MOSCAT 
- MOSCAT: temporal_quality=jaccard_score, snapshot_quality=mse
- evol clustering: hc (default, from paper), snapshot_quality= mse like(default, from paper)
- Issue: because of different measures for tq the results are not comparable regarding hc or jaccard_score
    - But regarding snapshot quality
    - Solution: moscat: use 1-hc instead of jaccard score for temporal quality, see experiment_5
