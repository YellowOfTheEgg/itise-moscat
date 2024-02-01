from .Stage import Stage
from .Subplot import Subplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    def __init__(
        self,
        df,
        df_mapping=dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature1",
            f2_col="feature2",
            group_col="cluster_id",
        ),
        plot_settings=dict(col_wrap=3, bbox_to_anchor=(-1.2, -0.4)),
    ):
        self.df = df
        self.df_mapping = df_mapping
        self.plot_settings = plot_settings

        mpl.rc("font", family="serif", serif="CMU Serif")

    def generate_fig(self):
        stage = Stage(
            df=self.df, df_mapping=self.df_mapping, plot_settings=self.plot_settings
        )
        subplot = Subplot(df_mapping=self.df_mapping)
        g = stage.getStage()
        g = subplot.addSubplots(g)
        return plt
