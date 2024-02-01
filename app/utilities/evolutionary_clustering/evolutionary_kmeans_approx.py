import numpy as np
from scipy.spatial.distance import cdist
import numpy.lib.recfunctions
import warnings
import copy
from app.utilities.custom_kmeans.incremental_kmeans import IncrementalKmeans


class EvolApproxKmeans:
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
        snapshot_quality = np.sum(1 - dists.min(axis=0))
        return snapshot_quality

    def calc_hc(self, centroids, initial_centroids):
        hc = 0
        if len(initial_centroids) > 0:
            best_match_idx = cdist(centroids, initial_centroids).argmin(axis=1)
            sorted_init_centroids = initial_centroids[best_match_idx]
            hc = np.min(centroids - sorted_init_centroids)
            # dists = np.linalg.norm(centroids - sorted_init_centroids, axis=1)
            # hc = np.sum(dists)/len(dists)
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
            point_cloud = np.array(clustering[self.feature_cols].tolist())
            centroids_distances = np.sqrt(
                ((point_cloud - centroids[:, np.newaxis]) ** 2).sum(axis=2)
            )
            closest_distances = np.min(centroids_distances, axis=0)
            for i in range(0, self.n_clusters - len(used_cluster_ids)):
                farest_object_id = np.argmax(closest_distances)
                filler_centroid = point_cloud[farest_object_id]
                handled_centroids = np.concatenate(
                    (handled_centroids, [filler_centroid]), axis=0
                )
                handled_clustering[farest_object_id]["cluster_id"] = (
                    len(used_cluster_ids) + i
                )
                closest_distances[farest_object_id] = 0
        return handled_clustering, handled_centroids

    def get_filler(self, clustering):
        cluster_ids = np.unique(clustering["cluster_id"].tolist())
        filler_cluster_ids = []
        for cluster_id in cluster_ids:
            cluster_size = clustering[clustering["cluster_id"] == cluster_id].shape[0]
            if cluster_size == 1:
                filler_cluster_ids.append(cluster_id)
        return filler_cluster_ids

    def centroid_number_handler(self, clustering, centroids):
        used_cluster_ids = np.unique(clustering["cluster_id"])
        centroids = centroids[used_cluster_ids]
        handled_clustering = copy.deepcopy(clustering)
        handled_centroids = copy.deepcopy(centroids)

        if len(used_cluster_ids) < self.n_clusters:
            point_cloud = np.array(clustering[self.feature_cols].tolist())
            centroids_distances = np.sqrt(
                ((point_cloud - centroids[:, np.newaxis]) ** 2).sum(axis=2)
            )
            closest_distances = np.min(centroids_distances, axis=0)
            for i in range(0, self.n_clusters - len(used_cluster_ids)):
                farest_object_id = np.argmax(closest_distances)
                filler_centroid = point_cloud[farest_object_id]
                handled_centroids = np.concatenate(
                    (handled_centroids, [filler_centroid]), axis=0
                )
                handled_clustering[farest_object_id]["cluster_id"] = (
                    len(used_cluster_ids) + i
                )
                closest_distances[farest_object_id] = 0
        return handled_clustering, handled_centroids

    # def calc_gammas(self,current_centroid_weights, past_centroid_weights):
    def extract_centroid_weights(self, clustering, centroid_ids=None):
        if centroid_ids is None:
            centroid_ids = np.unique(clustering["cluster_id"])

        centroid_weights = []
        for centroid_id in centroid_ids:
            cluster = clustering[clustering["cluster_id"] == centroid_id]
            centroid_weights.append(cluster.shape[0])
        return np.array(centroid_weights)

    def get_filler_free_sorted_centroid_idxs(
        self,
        ff_current_centroid_idxs,
        ff_prev_centroid_idxs,
        current_centroids,
        prev_centroids,
    ):
        centroid_idxs_mapping = []
        for ff_current_centroid_idx in ff_current_centroid_idxs:
            distances = []
            for ff_prev_centroid_idx in ff_prev_centroid_idxs:
                c_centroid = current_centroids[ff_current_centroid_idx]
                p_centroid = prev_centroids[ff_prev_centroid_idx]
                distances.append(np.linalg.norm(c_centroid - p_centroid))
            centroid_idxs_mapping.append(np.argmin(distances))
        return centroid_idxs_mapping

    def fit(self, timeseries_data):
        times = np.unique(timeseries_data[self.time_col])
        final_centroids = []
        final_clusterings = []
        for i, t in enumerate(times):
            timepoint_data = timeseries_data[timeseries_data[self.time_col] == t]
            init_centroids = []
            if i > 0:
                init_centroids = final_centroids[i - 1]
            ikmeans = IncrementalKmeans(
                n_clusters=self.n_clusters,
                init_point_index=self.init_point_index,
                max_iter=self.max_iter,
                halflife=self.halflife,
                initial_centroids=init_centroids,
            )
            ikmeans.define_data(
                {
                    "object_id": self.object_id_col,
                    "time": self.time_col,
                    "features": self.feature_cols,
                }
            )
            ikmeans.fit(timepoint_data)
            current_centroids = ikmeans.output_parameters["centroids"]
            current_clustering = ikmeans.clustering_result
            if i == 0:
                final_centroids.append(current_centroids)
                final_clusterings.append(ikmeans.clustering_result)
            else:
                prev_filler_idx = self.get_filler(final_clusterings[i - 1])
                current_filler_idx = self.get_filler(current_clustering)
                ff_current_centroid_idxs = np.array(
                    list(
                        set(current_clustering["cluster_id"].tolist())
                        - set(current_filler_idx)
                    )
                )
                ff_prev_centroid_idxs = np.array(
                    list(
                        set(final_clusterings[i - 1]["cluster_id"].tolist())
                        - set(prev_filler_idx)
                    )
                )

                best_match_idx = self.get_filler_free_sorted_centroid_idxs(
                    ff_current_centroid_idxs,
                    ff_prev_centroid_idxs,
                    current_centroids,
                    final_centroids[i - 1],
                )
                past_centroid_weights = self.extract_centroid_weights(
                    final_clusterings[i - 1], best_match_idx
                )  # by best_match_idx list instead of clustering cause of order importance
                current_centroid_weights = self.extract_centroid_weights(
                    current_clustering, ff_current_centroid_idxs
                )
                gammas = current_centroid_weights / (
                    current_centroid_weights + past_centroid_weights
                )
                temporal_centroids = (
                    np.array([1 - gammas]).T
                    * self.cp
                    * final_centroids[i - 1][best_match_idx]
                    + np.array([gammas]).T
                    * (1 - self.cp)
                    * current_centroids[ff_current_centroid_idxs]
                )

                temporal_clustering = self._assign_items_to_centroids(
                    timepoint_data, temporal_centroids
                )

                handled_clustering, handled_centroids = self.centroid_number_handler(
                    temporal_clustering, temporal_centroids
                )

                final_centroids.append(handled_centroids)
                final_clusterings.append(handled_clustering)
        self.clusterings = final_clusterings
        self.centroids = final_centroids
