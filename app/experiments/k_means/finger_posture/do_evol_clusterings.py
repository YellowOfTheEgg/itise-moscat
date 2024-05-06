import numpy as np
from app.utilities.evolutionary_clustering.evolutionary_kmeans_inc import (
    EvolIncrementalKmeans,
)
import pandas as pd
import os
from app.utilities.custom_kmeans.custom_kmeans import Kmeans
import pickle5 as pickle
from scipy.spatial.distance import cdist


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


def do_evol_clustering(
    data, data_definition, init_point_index, k, cp, max_iter_set, halflife
):
    ekmeans = EvolIncrementalKmeans(
        k, init_point_index, max_iter=max(max_iter_set), cp=cp, halflife=halflife
    )
    ekmeans.define_data(data_definition)
    ekmeans.fit(data)
    return ekmeans


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


def extract_dict_from_stepset(stepset):
    stepset_dict = {
        "temporal_quality": stepset.temporal_quality,
        "snapshot_quality": stepset.snapshot_quality,
        "input_parameters": stepset.input_parameters,
        "output_parameters": stepset.output_parameters,
        "clustering": stepset.clustering,
    }
    return stepset_dict


if __name__ == "__main__":
    file_name = "ERing_FULL.csv"
    k_set = range(2, 11)
    max_iter_set = range(0, 21)
    weight_set = np.arange(0, 1.1, 0.1)
    # halflife=0.1
    halflife = 1

    pd.set_option("display.max_rows", None)
    abs_data_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../data/"
    data = load_data(abs_data_path + file_name)
    data_definition = {
        "object_id": "object_id",
        "time": "time",
        "features": ["feature_0", "feature_1", "feature_2", "feature_3"],
    }

    number_of_objects = len(np.unique(data["object_id"]))
    init_point_index_set = range(0, number_of_objects)

    evol_clusterings = []
    for k in k_set:
        init_point_index = extract_init_point_index(
            data, data_definition, init_point_index_set, k, max_iter=max(max_iter_set)
        )
        for w in weight_set:
            evol_clustering = do_evol_clustering(
                data,
                data_definition,
                init_point_index,
                k=k,
                cp=w,
                max_iter_set=max_iter_set,
                halflife=halflife,
            )
            clusterings = evol_clustering.clusterings
            centroids = evol_clustering.centroids
            parameter_sets = []
            for i in range(len(clusterings)):
                parameter_sets.append(
                    {
                        "clustering": clusterings[i],
                        "output_parameters": {"centroids": centroids[i]},
                    }
                )

            evol_clusterings.append({"k": k, "w": w, "parameter_sets": parameter_sets})
    output_dir = os.getcwd() + "/clusterings/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + f"evol.pickle", "wb") as f:
        pickle.dump(evol_clusterings, f)
