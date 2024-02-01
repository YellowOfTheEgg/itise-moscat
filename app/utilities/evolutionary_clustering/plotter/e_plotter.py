import pandas as pd
from app.utilities.visualisations.twoDClustering.Plotter import Plotter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class EPlotter:
    def __init__(self, e_kmeans):
        self.e_kmeans = e_kmeans

    def create_clustering_plot(self, img_path=""):
        timepoint_dfs = []
        for i, clustering in enumerate(self.e_kmeans.clusterings):
            df = pd.DataFrame.from_records(clustering)
            #            centroids_df = pd.DataFrame.from_records(self.e_kmeans.final_centroids[i])
            #            init_centroids_df =pd.DataFrame.from_records(self.e_kmeans.init_centroids[i])

            # print(centroids)

            #            centroids_df["cluster_id"] = -1
            #            init_centroids_df["cluster_id"] = -2
            #           df = pd.concat([clustering_df, centroids_df,init_centroids_df])
            timepoint_dfs.append(df)
        df = pd.concat(timepoint_dfs)

        df_mapping = dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature_1",
            f2_col="feature_2",
            group_col="cluster_id",
        )

        plotter = Plotter(df=df, df_mapping=df_mapping)
        fig = plotter.generate_fig()
        if img_path != "":
            fig.savefig(img_path)
        else:
            return fig

    def create_snapshot_quality_plot(self, path=""):
        number_of_objects = np.array(
            list(map(lambda l: len(l), self.e_kmeans.clusterings))
        )
        snapshot_quality_scores = (
            np.array(self.e_kmeans.snapshot_quality) / number_of_objects
        )
        sns.set_style("darkgrid")
        p = sns.lineplot(
            x=range(len(snapshot_quality_scores)), y=snapshot_quality_scores
        )
        p.set_ylim(0, 1)
        p.set(
            xlabel="Time", ylabel="Snapshot Quality", title="Snapshot Quality over Time"
        )
        if path != "":
            plt.savefig(path)
        else:
            return plt

    def create_history_cost_plot(self, path=""):
        # plt.clf()
        sns.set_style("darkgrid")
        p = sns.lineplot(
            x=range(len(self.e_kmeans.history_cost)), y=self.e_kmeans.history_cost
        )
        p.set_ylim(0, 1)
        p.set(xlabel="Time", ylabel="History Cost", title="Cost over Time")
        if path != "":
            plt.savefig(path)
        else:
            return plt
