import numpy as np
import statistics
import pandas as pd
import os
from tqdm import tqdm

import pickle5 as pickle
from sklearn import metrics


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


def extract_number_of_clusters(parameter_sets):
    number_of_clusters = []
    for parameter_set in parameter_sets:
        clustering = parameter_set["clustering"]
        number_of_clusters.append(len(np.unique(clustering["cluster_id"].tolist())))
    return number_of_clusters


def calc_purities(parameter_sets, label_mapping):
    def calc_purity(tp_clustering, label_mapping):
        clusters = tp_clustering["cluster_id"].tolist()
        object_ids = tp_clustering["object_id"].tolist()
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

    c_num = extract_number_of_clusters(parameter_sets)
    av_c_num = sum(c_num) / len(c_num)
    std_c_num = statistics.stdev(c_num)
    sqs = list(map(lambda p: p["snapshot_quality"] / 2, parameter_sets))
    tqs = list(map(lambda p: p["temporal_quality"], parameter_sets))
    purities = calc_purities(parameter_sets, label_mapping)

    av_sq = sum(sqs) / len(sqs)
    std_sq = statistics.stdev(sqs)
    av_tqs = sum(tqs) / len(tqs)
    std_tqs = statistics.stdev(tqs)
    # normalized_sqs=list(map(lambda sq: sq/2, sqs) )
    # total_score=(sum(normalized_sqs)+sum(tqs))/(len(normalized_sqs)+len(tqs))
    total_score = (sum(sqs) + sum(tqs)) / (len(sqs) + len(tqs))
    av_purity = sum(purities) / len(purities)
    std_purity = statistics.stdev(purities)
    return [
        av_c_num,
        std_c_num,
        av_sq,
        std_sq,
        av_tqs,
        std_tqs,
        total_score,
        av_purity,
        std_purity,
    ]


def run_evaluation(method_name, parameter_sets, data_definition, label_mapping):
    result = []
    for parameter_set in tqdm(parameter_sets, desc="Progress per file", leave=False):
        w = parameter_set["w"]

        # clusterings=parameter_set['parameter_sets'][0]['clustering
        stats = extract_clustering_stats(
            parameter_set["parameter_sets"], data_definition, label_mapping
        )
        result_row = [method_name, w] + stats
        result.append(result_row)

    result_df = pd.DataFrame(
        result,
        columns=[
            "method",
            "w",
            "av_c_num",
            "std_c_num",
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
    {"method": "baseline", "file_name": "baseline_dbscan.pickle"},
    {"method": "moscat", "file_name": "moscat_dbscan.pickle"},
    {"method": "evol", "file_name": "evol_dbscan.pickle"},
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
