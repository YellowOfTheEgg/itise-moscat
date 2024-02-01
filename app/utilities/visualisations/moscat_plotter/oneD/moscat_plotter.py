import pandas as pd
from app.utilities.visualisations.oneDClustering.Plotter import Plotter
from app.utilities.visualisations.moscat_plotter.pareto_plotters import ReferenceVectors
import os


class MoscatPlotter:
    def __init__(self, moscat):

        # data_description:
        # object_id_col: ''
        # time_col: ''
        # f1_col: ''
        # f2_col: ''
        # cluster_id_col: ''

        self.moscat = moscat
        self.object_id_col = self.moscat.col_object_id  # data_description.object_id_col
        self.time_col = self.moscat.col_time  # data_description.time_col
        self.features = self.moscat.col_features
        self.cluster_id_col = "cluster_id"  # data_description.cluster_id_col

    def create_optimal_clustering_plot(self, img_path):
        timepoint_dfs = []
        for optimal_parameter_set in self.moscat.optimal_parameters:
            df = pd.DataFrame.from_records(optimal_parameter_set.clustering)
            timepoint_dfs.append(df)
        df = pd.concat(timepoint_dfs)

        df_mapping = dict(
            time_col=self.time_col,
            object_id_col=self.object_id_col,
            f1_col=self.features[0],
            group_col=self.cluster_id_col,
        )

        plotter = Plotter(df=df, df_mapping=df_mapping)
        fig = plotter.generate_fig()
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        fig.savefig(img_path)

    def create_pareto_plots(self, img_path, strategy, **kwargs):
        if strategy == "reference_vectors":
            #  print(**kwargs)
            pareto_strategy = ReferenceVectors(self.moscat, **kwargs)
        pareto_strategy.create_plot(img_path=img_path)
