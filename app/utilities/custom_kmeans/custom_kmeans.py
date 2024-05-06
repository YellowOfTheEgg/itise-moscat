import random
import numpy as np
from scipy.spatial.distance import cdist
import numpy.lib.recfunctions


class Kmeans:
    def __init__(self, n_clusters, init_point_index=None, max_iter=None):
        self.n_clusters = n_clusters
        self.init_point_index = init_point_index
        self.max_iter = max_iter

        self.init_centroids = []
        self.final_centroids = []
        self.clustering_result = None

    # from k-means++

    def define_data(self, data_definition):
        self.object_id_col = data_definition["object_id"]
        self.time_col = data_definition["time"]
        self.feature_cols = data_definition["features"]

    def _init_centroids(self, timepoint_data):
        point_cloud = np.array(timepoint_data[self.feature_cols].tolist())
        centroids_pos = np.ndarray((self.n_clusters, len(self.feature_cols)))
        centroids_pos[0] = point_cloud[self.init_point_index]
        for i in range(1, self.n_clusters):
            dist = cdist(point_cloud, centroids_pos[:i]).min(axis=1)
            next_centroid = dist.argmax()
            centroids_pos[i] = point_cloud[next_centroid]
        return centroids_pos

    def _assign_items_to_centroids(self, timepoint_data, centroids):

        point_cloud = np.array(timepoint_data[self.feature_cols].tolist())
        centroids_distances = np.sqrt(
            ((point_cloud - centroids[:, np.newaxis]) ** 2).sum(axis=2)
        )
        closest_centroid_ids = np.argmin(centroids_distances, axis=0)
        clustered_data = np.lib.recfunctions.append_fields(
            timepoint_data,
            "cluster_id",
            closest_centroid_ids,
            dtypes="i4",
            usemask=False,
        )
        return clustered_data

    def _recalculate_centroids(self, clustered_timepoint_data):
        cluster_ids = np.unique(clustered_timepoint_data["cluster_id"])
        new_centroid_positions = [
            np.array(
                clustered_timepoint_data[
                    clustered_timepoint_data["cluster_id"] == c_id
                ][self.feature_cols].tolist()
            ).mean(axis=0)
            * 1
            for c_id in cluster_ids
        ]
        new_centroid_positions = np.array(new_centroid_positions)
        return new_centroid_positions

    def _termination_condition(self, prev_centroids, current_centroids):
        prev_centroid_positions = np.array(prev_centroids[self.feature_cols].tolist())
        current_centroid_positions = np.array(
            current_centroids[self.feature_cols].tolist()
        )
        return np.allclose(prev_centroid_positions, current_centroid_positions)

    def fit(self, timepoint_data):
        self.init_centroids = []
        self.final_centroids = []
        self.clustering_result = None
        centroids = self._init_centroids(timepoint_data)

        clustering_result = self._assign_items_to_centroids(timepoint_data, centroids)
        # new_centroids = self._recalculate_centroids(clustered_data)
        current_iter = 0
        terminate = False
        while not terminate and current_iter < self.max_iter:
            current_iter += 1
            new_centroids = self._recalculate_centroids(clustering_result)
            clustering_result = self._assign_items_to_centroids(
                timepoint_data, centroids
            )
            if np.allclose(centroids, new_centroids):
                terminate = True
            else:
                centroids = new_centroids
        self.final_centroids = centroids
        self.clustering_result = clustering_result
