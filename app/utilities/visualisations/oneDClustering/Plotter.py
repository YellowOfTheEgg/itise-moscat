import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from .Stage import Stage
from .LinePlot import LinePlot
from .ScatterPlot import ScatterPlot


class Plotter:
    def __init__(
        self,
        df,
        df_mapping=dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature1",
            group_col="cluster_id",
            outlier_col="outlier",
        ),
        outlier_method="sigma",
    ):
        self.df = df
        self.df_mapping = df_mapping

        self.outlier_method = outlier_method
        mpl.rc("font", family="serif", serif="CMU Serif")

    def generate_fig(self):
        stage = Stage()
        lineplot = LinePlot(
            df=self.df,
            df_mapping=self.df_mapping,
            stage=stage.getStage(),
            outlier_method=self.outlier_method,
        )
        scatterplot = ScatterPlot(df=self.df, df_mapping=self.df_mapping)
        lineplot.addLinePlots()
        scatterplot.addScatters()
        plt.tight_layout()
        return plt
