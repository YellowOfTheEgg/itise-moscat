from collections import defaultdict
from app.moscat.interfaces.snapshot_quality_measures.interface import IQualityMeasure
import numpy as np


class MSEScore(IQualityMeasure):

    lower_bound = 0
    upper_bound = 1

    def define_data(self, data_definition):
        self.col_object_id = data_definition["object_id"]
        self.col_time = data_definition["time"]
        self.col_features = data_definition["features"]

    def _calculate_mse(self, clustering):
        cluster_points = defaultdict(list)
        for row in clustering:
            cluster_points[row["cluster_id"]] += [(row["feature_1"], row["feature_2"])]
        squared_error = []
        for k, v in cluster_points.items():
            cluster_centroid = np.sum(v, axis=0) / len(v)
            squared_deviation = sum(
                map(
                    lambda x: ((x[0] - cluster_centroid[0]) ** 2)
                    + ((x[1] - cluster_centroid[1]) ** 2),
                    cluster_points[k],
                )
            )
            squared_error.append(squared_deviation / (len(cluster_points[k])))
        return sum(squared_error) / len(squared_error)

    def calculate_score(self, clustering):
        return 1 - self._calculate_mse(clustering)
