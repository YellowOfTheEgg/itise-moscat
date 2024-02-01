from collections import defaultdict
from app.moscat.interfaces.snapshot_quality_measures.interface import IQualityMeasure
import numpy as np


class EvolScore(IQualityMeasure):
    lower_bound = 0
    upper_bound = 1

    def define_data(self, data_definition):
        self.col_object_id = data_definition["object_id"]
        self.col_time = data_definition["time"]
        self.col_features = data_definition["features"]

    def _calculate_evol_score(self, clustering, output_parameters):
        cluster_points = defaultdict(list)
        for row in clustering:
            cluster_points[row["cluster_id"]] += [tuple(row[self.col_features])]

        centroids = output_parameters["centroids"]
        inv_deviatons = []

        for k, v in cluster_points.items():
            cluster_centroid = centroids[k]
            inv_deviatons += list(
                map(
                    lambda x: 1 - np.linalg.norm(cluster_centroid - x),
                    cluster_points[k],
                )
            )
        score = sum(inv_deviatons) / len(inv_deviatons)
        return score

    def calculate_score(self, clustering, output_parameters=None):
        evol_score = self._calculate_evol_score(clustering, output_parameters)
        return evol_score
