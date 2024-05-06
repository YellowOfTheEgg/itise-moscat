import numpy as np
from app.moscat.interfaces import clustering_methods
from app.moscat.moscat import Moscat
from app.moscat.interfaces import pareto_strategies
from app.moscat.interfaces import snapshot_quality_measures, temporal_quality_measures
import pandas as pd
import os
import pickle5 as pickle
from app.utilities.custom_kmeans.custom_kmeans import Kmeans
from scipy.spatial.distance import cdist


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


def extract_pareto_front(inter_steps):
    pareto_fronts = []
    for step in inter_steps:
        pareto_front = []
        for el in step["pareto_front"]:
            sq = el.snapshot_quality
            tq = el.temporal_quality
            pareto_front.append([sq, tq])
        pareto_fronts.append(pareto_front)
    output_dir = os.getcwd() + "/pareto_fronts/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + f"moscat.pickle", "wb") as f:
        pickle.dump(pareto_fronts, f)

    return


def do_moscat_clustering(
    data, data_definition, init_point_index, k_set, moscat_weight, max_iter_set
):
    clustering_method = clustering_methods.IIncrementalKmeans(
        k_set=k_set, max_iter_set=max_iter_set, init_point_index=init_point_index
    )
    snapshot_quality_measure = snapshot_quality_measures.EvolScore()
    temporal_quality_measure = temporal_quality_measures.EvolScore()
    pareto_selection_strategy = pareto_strategies.NextToReferenceVectors(
        max_snapshot_quality=1, max_temporal_quality=1, weight=moscat_weight
    )

    moscat = Moscat(
        clustering_method=clustering_method,
        snapshot_quality_measure=snapshot_quality_measure,
        temporal_quality_measure=temporal_quality_measure,
        pareto_selection_strategy=pareto_selection_strategy,
        log_intermediate_steps=True,
    )
    moscat.define_data(data_definition)
    moscat.calculate_optimal_parameters(data)

    #  intermed_steps=moscat.intermediate_steps
    # extract_pareto_front(intermed_steps)
    return moscat


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
    file_name = "BasicMotions_FULL.csv"
    k_set = range(2, 11)

    max_iter_set = range(1, 21)

    weight_set = np.arange(0, 1.1, 0.1)

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

    moscat_clusterings = []

    for k in k_set:
        init_point_index = extract_init_point_index(
            data, data_definition, init_point_index_set, k, max_iter=10
        )
        for w in weight_set:
            moscat = do_moscat_clustering(
                data,
                data_definition,
                init_point_index,
                k_set=[k],
                moscat_weight=1 - w,
                max_iter_set=max_iter_set,
            )

            parameter_sets = list(
                map(lambda p: extract_dict_from_stepset(p), moscat.optimal_parameters)
            )
            moscat_clusterings.append(
                {"k": k, "w": w, "parameter_sets": parameter_sets}
            )

    output_dir = os.getcwd() + "/clusterings/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + f"moscat.pickle", "wb") as f:
        pickle.dump(moscat_clusterings, f)
