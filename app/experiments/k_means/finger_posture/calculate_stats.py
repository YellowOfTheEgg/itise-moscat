import numpy as np
import statistics
import pandas as pd
import os
from tqdm import tqdm

import pickle5 as pickle
from sklearn import metrics

from app.moscat.interfaces import temporal_quality_measures
from app.moscat.interfaces import snapshot_quality_measures


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_data(data_path):
    data = np.genfromtxt(
        data_path,
        delimiter=",",
        skip_header=1,
        dtype=[
            ("object_id", "U10"),
            ("time", "i4"),
            ("feature_0", "d"),
            ("feature_1", "d"),
            ("feature_2", "d"),
            ("feature_3", "d"),
        ],
    )
    return data


def extract_centroids(tp_clustering, data_definition):
    centroid_map = {}  # key: cluster_id, value: centroid
    for cluster_id in np.unique(tp_clustering["cluster_id"]):
        cluster_data = tp_clustering[tp_clustering["cluster_id"] == cluster_id]
        cluster_points = cluster_data[data_definition["features"]]
        cluster_points = np.array(
            list(map(lambda tpl: np.array(list(tpl)), cluster_points))
        )
        centroid = cluster_points.mean(axis=0)
        centroid_map[cluster_id] = centroid

    centroids = np.array(
        list(
            map(
                lambda cluster_id: centroid_map[cluster_id], tp_clustering["cluster_id"]
            )
        )
    )
    return centroids


def calc_snapshot_qualities(parameter_sets, data_definition):

    snapshot_qualities = []
    evol_sq = snapshot_quality_measures.EvolScore()
    evol_sq.define_data(data_definition)

    for parameter_set in parameter_sets:
        clustering = parameter_set["clustering"]
        output_parameters = parameter_set["output_parameters"]

        sq = evol_sq.calculate_score(clustering, output_parameters)
        snapshot_qualities.append(sq)
    return snapshot_qualities


def calc_temporal_qualites(parameter_sets, data_definition):
    temporal_qualities = []
    evol_tq = temporal_quality_measures.EvolScore()
    evol_tq.define_data(data_definition)
    for i, parameter_set in enumerate(parameter_sets):
        if i == 0:
            temporal_qualities.append(0)
        else:
            past_clustering_info = AttrDict(parameter_sets[i - 1])
            current_clustering = parameter_set["clustering"]
            current_output_parameters = parameter_set["output_parameters"]

            tq = evol_tq.calculate_score(
                past_clustering_info, current_clustering, current_output_parameters
            )
            temporal_qualities.append(tq)
    return temporal_qualities


def get_filler(clustering):
    cluster_ids = np.unique(clustering["cluster_id"].tolist())
    filler_cluster_ids = []
    for cluster_id in cluster_ids:
        cluster_size = clustering[clustering["cluster_id"] == cluster_id].shape[0]
        if cluster_size == 1:
            filler_cluster_ids.append(cluster_id)
    return filler_cluster_ids


def calc_purities(parameter_sets, label_mapping):
    def calc_purity(tp_clustering, label_mapping):
        filler_cluster_ids = get_filler(tp_clustering)
        filler_free_tp_clustering = tp_clustering[
            ~np.isin(tp_clustering["cluster_id"], filler_cluster_ids)
        ]
        clusters = filler_free_tp_clustering["cluster_id"].tolist()
        object_ids = filler_free_tp_clustering["object_id"].tolist()
        labels = list(map(lambda o: label_mapping[o], object_ids))
        confusion_matrix = metrics.cluster.contingency_matrix(labels, clusters)
        purity = np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
        return purity

    purities = []
    for parameter_set in parameter_sets:
        purity = calc_purity(parameter_set["clustering"], label_mapping)
        purities.append(purity)
    return purities


def extract_clustering_stats(parameter_sets, data_definition, label_mapping):

    sqs = calc_snapshot_qualities(parameter_sets, data_definition)
    tqs = calc_temporal_qualites(parameter_sets, data_definition)
    purities = calc_purities(parameter_sets, label_mapping)

    av_sq = sum(sqs) / len(sqs)
    std_sq = statistics.stdev(sqs)
    av_tqs = sum(tqs) / len(tqs)
    std_tqs = statistics.stdev(tqs)
    total_score = (sum(sqs) + sum(tqs)) / (len(sqs) + len(tqs))
    av_purity = sum(purities) / len(purities)
    std_purity = statistics.stdev(purities)
    return [av_sq, std_sq, av_tqs, std_tqs, total_score, av_purity, std_purity]


def run_evaluation(method_name, parameter_sets, data_definition, label_mapping):
    result = []
    for parameter_set in tqdm(parameter_sets, desc="Progress per file", leave=False):
        w = parameter_set["w"]
        k = parameter_set["k"]
        # clusterings=parameter_set['parameter_sets'][0]['clustering
        stats = extract_clustering_stats(
            parameter_set["parameter_sets"], data_definition, label_mapping
        )
        result_row = [method_name, k, w] + stats
        result.append(result_row)

    result_df = pd.DataFrame(
        result,
        columns=[
            "method",
            "k",
            "w",
            "av_sq",
            "std_sq",
            "av_tq",
            "std_tq",
            "av_total_score",
            "av_purity",
            "std_purity",
        ],
    )
    return result_df


file_settings = [
    {"method": "baseline", "file_name": "baseline.pickle"},
    {"method": "moscat", "file_name": "moscat.pickle"},
    # {'method':'moscatJJ', 'file_name': 'moscatJJ.pickle'},
    {"method": "evol", "file_name": "evol.pickle"},
    # {'method':'evol_approx', 'file_name': 'evol_approx.pickle'}
]

if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    abs_çlustering_path = os.path.dirname(os.path.abspath(__file__)) + "/clusterings/"
    abs_data_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../data/"
    data_definition = {
        "object_id": "object_id",
        "time": "time",
        "features": ["feature_0", "feature_1", "feature_2", "feature_3"],
    }

    data = pd.read_csv(abs_data_path + "ERing_FULL.csv")
    data["object_id"] = data["object_id"].astype("string")
    label_mapping = pd.Series(data["label"].values, index=data["object_id"]).to_dict()

    merged_evaluations = []
    for file_setting in tqdm(file_settings, desc="Total Progress"):
        parameters_file = open(abs_çlustering_path + file_setting["file_name"], "rb")
        parameter_sets = pickle.load(parameters_file)
        evaluation_df = run_evaluation(
            file_setting["method"], parameter_sets, data_definition, label_mapping
        )
        merged_evaluations.append(evaluation_df)

    merged_evaluations_df = pd.concat(merged_evaluations)

    output_dir = os.getcwd() + "/stats/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    merged_evaluations_df.to_csv(output_dir + f"evaluation_result.csv", index=False)
