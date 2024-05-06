from app.moscat.interfaces.temporal_quality_measures.interface import (
    ITemporalQualityMeasure,
)
import itertools
import numpy as np


class JaccardScore(ITemporalQualityMeasure):
    lower_bound = 0
    upper_bound = 1

    def define_data(self, data_definition):
        self.col_object_id = data_definition["object_id"]
        self.col_time = data_definition["time"]
        self.col_features = data_definition["features"]

    def _calculate_jaccard(self, clustering_1, clustering_2):
        groups_1 = [
            list(v)
            for k, v in itertools.groupby(clustering_1, lambda row: row["cluster_id"])
        ]
        groups_1 = list(
            map(lambda group: list(map(lambda row: row["object_id"], group)), groups_1)
        )
        groups_2 = [
            list(v)
            for k, v in itertools.groupby(clustering_2, lambda row: row["cluster_id"])
        ]
        groups_2 = list(
            map(lambda group: list(map(lambda row: row["object_id"], group)), groups_2)
        )

        unique_transition_weights = []
        for group_1 in groups_1:
            for group_2 in groups_2:
                intersec_len = len(list(set(group_1) & set(group_2)))
                if (
                    intersec_len
                    > 0 & intersec_len
                    < min(len(set(group_1)), len(set(group_2)))
                ):
                    union_len = len(list(set(group_1) | set(group_2)))
                    transition_weight = intersec_len / union_len
                    unique_transition_weights.append(transition_weight)
        transition_score = sum(unique_transition_weights) / len(
            unique_transition_weights
        )
        transition_score = round(transition_score, 2)
        return transition_score

    def remove_filler(self, clustering):
        cluster_ids = np.unique(clustering["cluster_id"])
        filler_cluster_ids = []
        for cluster_id in cluster_ids:
            cluster_size = clustering[clustering["cluster_id"] == cluster_id].shape[0]
            if cluster_size == 1:
                filler_cluster_ids.append(cluster_id)
        clustering = clustering[~np.isin(clustering["cluster_id"], filler_cluster_ids)]
        return clustering

    def calculate_score(
        self, past_clustering_info, current_clustering, current_output_parameters
    ):
        # ff_current_clustering=self.remove_filler(current_clustering)# added for eval
        # ff_past_clustering=self.remove_filler(past_clustering_info.clustering)# added for eval
        return self._calculate_jaccard(
            past_clustering_info.clustering, current_clustering
        )
