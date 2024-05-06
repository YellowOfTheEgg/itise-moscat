import numpy as np
from app.moscat.interfaces import clustering_methods
from app.moscat.moscat import Moscat
from app.moscat.interfaces import pareto_strategies
from app.moscat.interfaces import snapshot_quality_measures, temporal_quality_measures
import pandas as pd
import os
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


def do_moscat_clustering(data, data_definition, eps_range, minpts_range, w):
    clustering_method = clustering_methods.DBSCAN(
        eps_range=eps_range, minpts_range=minpts_range
    )
    snapshot_quality_measure = snapshot_quality_measures.SilhouetteCoefficient()
    temporal_quality_measure = temporal_quality_measures.JaccardScore()
    pareto_selection_strategy = pareto_strategies.NextToReferenceVectors(
        max_snapshot_quality=snapshot_quality_measure.upper_bound,
        max_temporal_quality=1,
        weight=1 - w,
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
    pd.set_option("display.max_rows", None)
    abs_data_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../data/"
    data = load_data(abs_data_path + "BasicMotions_FULL.csv")
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

    eps_range = [0.1, 1, 0.1]
    minpts_range = [2, 10, 1]
    weight_set = np.arange(0.0, 1.1, 0.1)
    moscat_clusterings = []

    moscat_clusterings = []
    for w in weight_set:
        moscat = do_moscat_clustering(data, data_definition, eps_range, minpts_range, w)
        parameter_sets = list(
            map(lambda p: extract_dict_from_stepset(p), moscat.optimal_parameters)
        )
        moscat_clusterings.append({"w": w, "parameter_sets": parameter_sets})

    

    output_dir = os.getcwd() + "/clusterings/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + f"moscat_dbscan.pickle", "wb") as f:
        pickle.dump(moscat_clusterings, f)
