from app.moscat.interfaces.temporal_quality_measures.interface import (
    ITemporalQualityMeasure,
)
import numpy as np
from scipy.spatial.distance import cdist


class EvolScore(ITemporalQualityMeasure):
    lower_bound = 0
    upper_bound = 1

    def define_data(self, data_definition):
        self.col_object_id = data_definition["object_id"]
        self.col_time = data_definition["time"]
        self.col_features = data_definition["features"]

    def _extract_centroids(self, tp_clustering, feature_names):
        centroid_map = {}  # key: cluster_id, value: centroid
        for cluster_id in np.unique(tp_clustering["cluster_id"]):
            cluster_data = tp_clustering[tp_clustering["cluster_id"] == cluster_id]
            cluster_points = cluster_data[feature_names]
            cluster_points = np.array(
                list(map(lambda tpl: np.array(list(tpl)), cluster_points))
            )
            centroid = cluster_points.mean(axis=0)
            centroid_map[cluster_id] = centroid

        centroids = np.array(
            list(
                map(
                    lambda cluster_id: centroid_map[cluster_id],
                    tp_clustering["cluster_id"],
                )
            )
        )
        return centroids

    def _sort_prev_centroids(self, prev_centroids, current_centroids):
        best_match_idx = cdist(current_centroids, prev_centroids).argmin(axis=1)
        return prev_centroids[best_match_idx]

    def remove_centroid_filler(self, centroids, clustering):
        cleaned_centroids = []
        for i, centroid in enumerate(centroids):
            cluster_size = clustering[clustering["cluster_id"] == i].shape[0]
            if cluster_size > 1:
                cleaned_centroids.append(centroid)
        return np.array(cleaned_centroids)

    def _calculate_evol_temporal_score(
        self, past_clustering_info, current_clustering, current_output_parameters
    ):
        past_centroids_with_filler = past_clustering_info.output_parameters["centroids"]
        current_centroids_with_filler = current_output_parameters["centroids"]
        # past_centroids=self.remove_centroid_filler(past_centroids_with_filler,past_clustering_info.clustering)
        # current_centroids=self.remove_centroid_filler(current_centroids_with_filler,current_clustering)

        past_centroids = past_centroids_with_filler
        current_centroids = current_centroids_with_filler
        reduced_past_centroids = past_centroids
        reduced_current_centroids = current_centroids
        if len(past_centroids) > len(current_centroids):
            best_match_idx = cdist(current_centroids, past_centroids).argmin(axis=1)
            reduced_past_centroids = past_centroids[best_match_idx]
        elif len(past_centroids) < len(current_centroids):
            best_match_idx = cdist(past_centroids, current_centroids).argmin(axis=1)
            reduced_current_centroids = current_centroids[best_match_idx]

        sorted_prev_centroids = self._sort_prev_centroids(
            reduced_past_centroids, reduced_current_centroids
        )
        dists = np.linalg.norm(
            reduced_current_centroids - sorted_prev_centroids, axis=1
        )
        hc = np.sum(dists) / len(dists)  # np.min(dists)
        return 1 - hc

    def calculate_score(
        self, past_clustering_info, current_clustering, current_output_parameters
    ):
        return self._calculate_evol_temporal_score(
            past_clustering_info, current_clustering, current_output_parameters
        )
