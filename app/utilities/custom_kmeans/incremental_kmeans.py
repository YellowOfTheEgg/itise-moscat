import numpy as np
from scipy.spatial.distance import cdist
import numpy.lib.recfunctions
import warnings
import copy


class IncrementalKmeans:
    def __init__(
        self,
        n_clusters,
        init_point_index=None,
        max_iter=None,
        halflife=1,
        initial_centroids=[],
        handle_centroid_number=True,
    ):
        self.n_clusters = n_clusters
        self.init_point_index = init_point_index
        self.max_iter = max_iter
        self.halflife = halflife
        self.initial_centroids = initial_centroids
        #  self.final_centroids = []
        self.clustering_result = None
        self.output_parameters = {"centroids": []}
        self.handle_centroid_number = handle_centroid_number

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

        clustered_data = numpy.lib.recfunctions.append_fields(
            timepoint_data,
            "cluster_id",
            closest_centroid_ids,
            dtypes="i4",
            usemask=False,
        )
        return clustered_data

    def _recalculate_centroids(self, clustered_timepoint_data):
        # cluster_ids=range(0,self.n_clusters)
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

    # todo: check and restructure the function. Reason: could be buggy in terms of logical errors
    def centroid_number_handler_save(self, clustering, centroids):
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

    def centroid_number_handler(self, clustering, centroids):

        # centroids = centroids[used_cluster_ids]
        handled_clustering = copy.deepcopy(clustering)
        handled_centroids = copy.deepcopy(centroids)

        if centroids.shape[0] < self.n_clusters:
            handled_clustering = copy.deepcopy(clustering)
            handled_centroids = copy.deepcopy(centroids)
            used_cluster_ids = np.unique(clustering["cluster_id"])
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

    def fit(self, timepoint_data):
        # centroids=[]
        if len(self.initial_centroids) == 0:
            centroids = self._init_centroids(timepoint_data)
        else:
            centroids = self.initial_centroids

        clustering_result = self._assign_items_to_centroids(timepoint_data, centroids)

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
        #    if self.handle_centroid_number:
        #        handled_clustering,handled_centroids=self.centroid_number_handler(clustering_result,centroids)
        #    else:
        #        handled_clustering,handled_centroids=clustering_result,centroids
        self.output_parameters["centroids"] = centroids
        self.clustering_result = clustering_result
