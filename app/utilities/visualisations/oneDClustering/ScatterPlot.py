import matplotlib.pyplot as plt


class ScatterPlot:
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
    ):
        self.df = df
        self.scatter_palette = [
            "#1E88E5",
            "#FFC107",
            "#1CF2AA",
            "#81019B",
            "#92C135",
            "#9C9C6E",
            "#1033EA",
            "#55A1D7",
            "#38CA48",
            "#637645",
            "#E9E2A3",
            "#F0A054",
            "#1E88E5",
            "#FFC107",
            "#1CF2AA",
            "#81019B",
        ]
        self.noise_color = "#D81B60"
        self.df_mapping = df_mapping
        self.color_counter = 0

    def get_color(self, cluster_id):
        color = ""
        #  if cluster_id==3:
        #      color='black'
        #      self.color_counter += 1
        if cluster_id > -1:
            color_id = self.color_counter % len(self.scatter_palette)
            color = self.scatter_palette[color_id]
            self.color_counter += 1
        else:
            color = self.noise_color
        return color

    def addScatters(self):
        group_col = self.df_mapping["group_col"]
        time_col = self.df_mapping["time_col"]
        feature_col = self.df_mapping["f1_col"]
        cluster_ids = self.df[group_col].unique()
        for cluster_id in cluster_ids:
            color = self.get_color(cluster_id)
            df_by_cluster = self.df.loc[self.df[group_col] == cluster_id]
            for index, row in df_by_cluster.iterrows():
                plt.scatter(
                    x=row[time_col],
                    y=row[feature_col],
                    c=color,
                    marker="o",
                    s=2,
                    zorder=3,
                )
