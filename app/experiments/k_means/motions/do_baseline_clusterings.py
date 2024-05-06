import numpy as np
import pandas as pd
import os
from app.utilities.custom_kmeans.custom_kmeans import Kmeans
from app.utilities.evolutionary_clustering.evolutionary_kmeans_inc import (
    EvolIncrementalKmeans,
)
from scipy.spatial.distance import cdist
import pickle5 as pickle


def load_data(data_path):
    data = np.genfromtxt(
        data_path,
        delimiter=",",
        skip_header=1,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7],
        dtype=[
            ("object_id", "U10"),
            ("time", "i4"),
            ("feature_0", "d"),
            ("feature_1", "d"),
            ("feature_2", "d"),
            ("feature_3", "d"),
            ("feature_4", "d"),
            ("feature_5", "d"),
        ],
    )
    return data


def calculate_sq(centroids, timepoint_data, feature_cols):
    clustered_points = np.array(timepoint_data[feature_cols].tolist())
    dists = cdist(centroids, clustered_points)
    snapshot_quality = np.sum(1 - dists.min(axis=0)) / len(clustered_points)
    return snapshot_quality


def extract_init_point_index(
    timepoint_data, data_definition, init_point_index_set, n_clusters, max_iter=10
):
    sqs = []
    for init_point_index in init_point_index_set:
        kmeans = Kmeans(n_clusters, init_point_index, max_iter)
        kmeans.define_data(data_definition)
        kmeans.fit(timepoint_data)
        centroids = kmeans.final_centroids
        sq = calculate_sq(centroids, timepoint_data, kmeans.feature_cols)
        sqs.append(sq)
    max_sq_idx = np.argmax(sqs)
    return max_sq_idx


def do_baseline_clustering(data, data_definition, init_point_index, k, max_iter_set):
    ekmeans = EvolIncrementalKmeans(
        k, init_point_index, max_iter=max(max_iter_set), cp=0
    )
    ekmeans.define_data(data_definition)
    ekmeans.fit(data)
    return ekmeans.clusterings, ekmeans.centroids


if __name__ == "__main__":
    file_name = "BasicMotions_FULL.csv"
    k_set = range(2, 11)
    max_iter_set = range(0, 21)

    pd.set_option("display.max_rows", None)
    abs_data_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../data/"
    data = load_data(abs_data_path + file_name)

    data_definition = {
        "object_id": "object_id",
        "time": "time",
        "features": [
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ],
    }

    number_of_objects = len(np.unique(data["object_id"]))
    init_point_index_set = range(0, number_of_objects)

    baseline_clusterings = []

    for k in k_set:
        init_point_index = extract_init_point_index(
            data, data_definition, init_point_index_set, k, max_iter=max(max_iter_set)
        )
        clusterings, centroids = do_baseline_clustering(
            data, data_definition, init_point_index, k, max_iter_set
        )
        parameter_sets = []
        for i in range(len(clusterings)):
            parameter_sets.append(
                {
                    "clustering": clusterings[i],
                    "output_parameters": {"centroids": centroids[i]},
                }
            )
        baseline_clusterings.append({"k": k, "w": 0, "parameter_sets": parameter_sets})

    output_dir = os.getcwd() + "/clusterings/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + f"baseline.pickle", "wb") as f:
        pickle.dump(baseline_clusterings, f)
