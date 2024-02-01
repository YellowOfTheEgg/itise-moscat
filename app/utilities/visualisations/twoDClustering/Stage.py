import seaborn as sns


class Stage:
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
        self.col_wrap = plot_settings["col_wrap"]
        self.time_col = df_mapping["time_col"]
        self.group_col = df_mapping["group_col"]

    def getStage(self):
        # sns.set(style="darkgrid", font="CMU Serif", font_scale=1.4)
        sns.set(style="darkgrid", font="Arial", font_scale=1.4)

        color_palette = sns.color_palette(palette="Set1")
        outlier_color = (0.8941176470588236, 0.10196078431372549, 0.10980392156862745)
        color_palette.remove(outlier_color)
        g = sns.FacetGrid(
            col_wrap=self.col_wrap,
            data=self.df,
            col=self.time_col,
            palette=color_palette,
            hue=self.group_col,
        )
        return g
