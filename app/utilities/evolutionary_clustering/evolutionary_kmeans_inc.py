import numpy as np
from scipy.spatial.distance import cdist
import numpy.lib.recfunctions
import warnings
import copy


class EvolIncrementalKmeans:
    def __init__(
        self, n_clusters, init_point_index=None, max_iter=None, cp=0.1, halflife=1
    ):
        self.n_clusters = n_clusters
        self.init_point_index = init_point_index
        self.max_iter = max_iter
        self.halflife = halflife
        self.cp = cp
        self.clusterings = []
        self.centroids = []
        self.oqs = []

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

    def _assign_items_to_centroids(self, timepoint_data, current_centroids):
        point_cloud = np.array(timepoint_data[self.feature_cols].tolist())
        centroids_distances = cdist(current_centroids, point_cloud)

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            new_centroid_positions = [
                np.array(
                    clustered_timepoint_data[
                        clustered_timepoint_data["cluster_id"] == c_id
                    ][self.feature_cols].tolist()
                ).mean(axis=0)
                for c_id in cluster_ids
            ]

        new_centroid_positions = np.array(new_centroid_positions)
        return new_centroid_positions

    def calculate_sq(self, centroids, timepoint_data):
        clustered_points = np.array(timepoint_data[self.feature_cols].tolist())
        dists = cdist(centroids, clustered_points)
        snapshot_quality = np.sum(1 - dists.min(axis=0)) / len(clustered_points)
        return snapshot_quality

    def calc_hc(self, centroids, initial_centroids):
        hc = 0
        if len(initial_centroids) > 0:
            best_match_idx = cdist(centroids, initial_centroids).argmin(axis=1)
            sorted_init_centroids = initial_centroids[best_match_idx]
            dists = np.linalg.norm(centroids - sorted_init_centroids, axis=1)
            hc = np.sum(dists) / len(dists)
        return hc

    def calc_overall_quality(self, centroids, initial_centroids, timepoint_data):
        sq = self.calculate_sq(centroids, timepoint_data)
        hc = self.calc_hc(centroids, initial_centroids)
        return sq - self.cp * hc

    def centroid_number_handler(self, clustering, centroids):
        used_cluster_ids = np.unique(clustering["cluster_id"])
        centroids = centroids[used_cluster_ids]
        handled_clustering = copy.deepcopy(clustering)
        handled_centroids = copy.deepcopy(centroids)

        if len(used_cluster_ids) < self.n_clusters:
            unique, counts = np.unique(
                clustering["cluster_id"].tolist(), return_counts=True
            )
            mask = np.array(
                list(
                    map(
                        lambda c: 1 if c in unique[counts > 1] else 0,
                        clustering["cluster_id"],
                    )
                )
            )
            point_cloud = np.array(clustering[self.feature_cols].tolist())
            centroids_distances = np.sqrt(
                ((point_cloud - centroids[:, np.newaxis]) ** 2).sum(axis=2)
            )
            closest_distances = np.min(centroids_distances, axis=0)
            closest_distances = closest_distances * mask

            for i in range(0, self.n_clusters - len(used_cluster_ids)):
                farest_object_id = np.argmax(closest_distances)
                filler_centroid = point_cloud[farest_object_id]
                handled_centroids = np.concatenate(
                    (handled_centroids, [filler_centroid]), axis=0
                )
                filler_centroid_id = len(used_cluster_ids) + i
                handled_clustering[farest_object_id]["cluster_id"] = filler_centroid_id
                closest_distances[farest_object_id] = 0

        return handled_clustering, handled_centroids

    def fit_timepoint_data(self, timepoint_data, initial_centroids=[]):

        intermediate_overall_scores = []
        intermediate_clusterings = []
        intermediate_centroids = []
        centroids = []
        if len(initial_centroids) == 0:
            centroids = self._init_centroids(timepoint_data)
        else:
            centroids = initial_centroids

        clustering_result = self._assign_items_to_centroids(timepoint_data, centroids)
        overall_quality = self.calc_overall_quality(
            centroids, centroids, timepoint_data
        )
        intermediate_overall_scores.append(overall_quality)
        intermediate_centroids.append(centroids)
        intermediate_clusterings.append(clustering_result)
        terminate = False
        current_iter = 1

        while not terminate and current_iter < self.max_iter:
            current_iter += 1
            new_centroids = self._recalculate_centroids(clustering_result)
            if len(centroids) != len(new_centroids):
                best_match_idx = cdist(new_centroids, centroids).argmin(axis=1)
                centroids = centroids[best_match_idx].copy()

            halflife_centroids = centroids + (new_centroids - centroids) * self.halflife

            clustering_result = self._assign_items_to_centroids(
                timepoint_data, halflife_centroids
            )
            if np.allclose(halflife_centroids, new_centroids) and np.allclose(
                centroids, halflife_centroids
            ):
                terminate = True
            else:
                centroids = halflife_centroids
                overall_quality = self.calc_overall_quality(
                    halflife_centroids, initial_centroids, timepoint_data
                )
                intermediate_overall_scores.append(overall_quality)
                # handled_clustering,handled_centroids=self.centroid_number_handler(clustering_result,halflife_centroids)
                intermediate_clusterings.append(clustering_result)
                intermediate_centroids.append(halflife_centroids)

        max_score_id = np.argmax(intermediate_overall_scores)
        max_oq_clustering = intermediate_clusterings[max_score_id]
        max_oq_centroids = intermediate_centroids[max_score_id]

        return max_oq_clustering, max_oq_centroids, max(intermediate_overall_scores)

    def fit(self, timeseries_data):
        times = np.unique(timeseries_data[self.time_col])
        for i, t in enumerate(times):
            timepoint_data = timeseries_data[timeseries_data[self.time_col] == t]
            init_centroids = []
            if i > 0:
                init_centroids = self.centroids[i - 1]
            max_oq_clustering, max_oq_centroids, max_oq = self.fit_timepoint_data(
                timepoint_data, init_centroids
            )
            self.clusterings.append(max_oq_clustering)
            self.centroids.append(max_oq_centroids)
            self.oqs.append(max_oq)
