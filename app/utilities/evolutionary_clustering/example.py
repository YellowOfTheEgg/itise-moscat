import os
import numpy as np
from app.utilities.evolutionary_clustering.evolutionary_kmeans_text import EKmeans

import pandas as pd
from app.utilities.evolutionary_clustering.plotter.e_plotter import EPlotter


def load_data():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = ROOT_DIR + "/../../data/generated_data_outlier.csv"
    data = np.genfromtxt(
        data_path,
        delimiter=",",
        skip_header=1,
        names=["object_id", "time", "feature_1", "feature_2"],
        dtype="U50,i4,f4,f4",
        usemask=False,
    )
    return data


def create_optimal_clustering_plot(clusterings, centroids, init_centroids, img_path):
    timepoint_dfs = []
    for i, clustering in enumerate(clusterings):
        clustering_df = pd.DataFrame.from_records(clustering)
        centroids_df = pd.DataFrame.from_records(centroids[i])
        init_centroids_df = pd.DataFrame.from_records(init_centroids[i])

        # print(centroids)

        centroids_df["cluster_id"] = -1
        init_centroids_df["cluster_id"] = -2
        df = pd.concat([clustering_df, centroids_df, init_centroids_df])
        timepoint_dfs.append(df)
    df = pd.concat(timepoint_dfs)

    df_mapping = dict(
        time_col="time",
        object_id_col="object_id",
        f1_col="feature_1",
        f2_col="feature_2",
        group_col="cluster_id",
    )

    plotter = EPlotter(df=df, df_mapping=df_mapping)
    fig = plotter.generate_fig()
    fig.savefig(img_path)


data = load_data()


number_of_objects = len(np.unique(data["object_id"]))


cp = 0
e_kmeans = EKmeans(n_centroids=3, cp=cp, n_repeats=1)
e_kmeans.fit(data, features=["feature_1", "feature_2"])
e_plotter = EPlotter(e_kmeans)
e_plotter.create_clustering_plot("evol.png")
# e_plotter.create_snapshot_quality_plot(f'sq_{cp}.png')
# e_plotter.create_history_cost_plot(f'hc_{cp}.png')
# print(e_kmeans.clustering_result)
# print(e_kmeans.snapshot_quality)
# create_optimal_clustering_plot(e_kmeans.clustering_result, e_kmeans.final_centroids,e_kmeans.init_centroids,'evol.png')
