import numpy as np
from app.utilities.evolutionary_clustering.evolutionary_dbscan import EvolDBSCAN
import pandas as pd
import os
import pickle5 as pickle


def load_data(data_path):
    data = np.genfromtxt(
        data_path,
        delimiter=",",
        skip_header=1,
        usecols=[0, 1, 2, 3, 4, 5],
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


def do_evol_clustering(data, data_definition, eps_set, min_pts_set):
    evol_dbscan = EvolDBSCAN(eps_set, min_pts_set)
    evol_dbscan.define_data(data_definition)
    evol_dbscan.fit(data)
    return evol_dbscan


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
    pd.set_option("display.max_rows", None)
    abs_data_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../data/"
    data = load_data(abs_data_path + "ERing_FULL.csv")
    data_definition = {
        "object_id": "object_id",
        "time": "time",
        "features": ["feature_0", "feature_1", "feature_2", "feature_3"],
    }

    eps_set = np.arange(0.1, 1.1, 0.1)
    min_pts_set = np.arange(2, 13, 1)
    weight_set = np.arange(0.0, 1.1, 0.1)

    evol_clustering = do_evol_clustering(data, data_definition, eps_set, min_pts_set)

    weighted_clusterings = []
    for w in weight_set:
        clustering_over_time = evol_clustering.select_clustering_over_time(w)

        row = {"w": w, "parameter_sets": clustering_over_time}
        weighted_clusterings.append(row)

    output_dir = os.getcwd() + "/clusterings/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + f"evol_dbscan.pickle", "wb") as f:
        pickle.dump(weighted_clusterings, f)
