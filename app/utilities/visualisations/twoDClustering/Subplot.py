import seaborn as sns
from .PointStyle import PointStyle


class Subplot:
    def __init__(
        self,
        df_mapping=dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature1",
            f2_col="feature2",
            group_col="cluster_id",
        ),
    ):
        self.f1_col = df_mapping["f1_col"]
        self.f2_col = df_mapping["f2_col"]
        self.group_col = df_mapping["group_col"]
        self.object_id_col = df_mapping["object_id_col"]

    def get_style_ts_patch(self, color):
        style_ts_patch = dict(
            fontsize=10,
            xytext=(-2, -1.5),
            textcoords="offset points",
            bbox=dict(boxstyle="square", alpha=0.1, color=color),
            va="top",
            ha="right",
            alpha=0.4,
        )
        return style_ts_patch

    def _get_sub_plot(self, x, y, z, **kwargs):
        point_style = PointStyle()
        ax = sns.scatterplot(
            x=x, y=y, **kwargs, marker="s", s=10
        )  # s=0 to hide scatter points
        number_of_points = len(x)

        for i in range(number_of_points):
            if int(kwargs["label"]) == -1:
                point_style.set_style(ax, i, x, y, z, kwargs, style_type="outlier")
            elif int(kwargs["label"]) == -2:
                point_style.set_style(ax, i, x, y, z, kwargs, style_type="preoutlier")
            else:
                point_style.set_style(ax, i, x, y, z, kwargs, style_type="default")
        return

    def addSubplots(self, g):

        g.map(self._get_sub_plot, self.f1_col, self.f2_col, self.object_id_col)
        return g
