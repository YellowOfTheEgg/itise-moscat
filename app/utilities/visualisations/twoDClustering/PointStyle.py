class PointStyle:
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

    def set_style(self, ax, i, x, y, z, kwargs, style_type="default"):
        if style_type == "default":
            ax.annotate(
                z.values[i],
                xy=(x.values[i], y.values[i]),
                fontsize=10,
                xytext=(-2, -1.5),
                textcoords="offset points",
                bbox=dict(boxstyle="square", alpha=0.1, color=kwargs["color"]),
                va="top",
                ha="right",
                alpha=0.4,
            )
        elif style_type == "outlier":
            ax.annotate(
                z.values[i],
                xy=(x.values[i], y.values[i]),
                fontsize=10,
                xytext=(-2, -1.5),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="square",
                    facecolor="red",
                    linestyle="",
                    linewidth=2,
                    alpha=0.1,
                ),
                va="top",
                ha="right",
                alpha=0,
                zorder=15,
            )
            ax.annotate(
                z.values[i],
                xy=(x.values[i], y.values[i]),
                fontsize=10,
                xytext=(-2, -1.5),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="square",
                    facecolor="none",
                    linestyle="--",
                    edgecolor="red",
                    linewidth=2,
                    alpha=1,
                ),
                va="top",
                ha="right",
                alpha=0,
                zorder=15,
            )
            ax.annotate(
                z.values[i],
                xy=(x.values[i], y.values[i]),
                fontsize=10,
                xytext=(-2, -1.5),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="square",
                    facecolor="none",
                    linestyle="--",
                    edgecolor="black",
                    linewidth=1,
                    alpha=1,
                ),
                va="top",
                ha="right",
                alpha=1,
                zorder=15,
            )

        elif style_type == "preoutlier":
            ax.annotate(
                z.values[i],
                xy=(x.values[i], y.values[i]),
                fontsize=10,
                xytext=(-2, -1.5),
                textcoords="offset points",
                bbox=dict(boxstyle="square", facecolor="gray", linestyle="", alpha=0.2),
                va="top",
                ha="right",
                alpha=0,
                zorder=10,
            )
            ax.annotate(
                z.values[i],
                xy=(x.values[i], y.values[i]),
                fontsize=10,
                xytext=(-2, -1.5),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="square",
                    facecolor="none",
                    linestyle="--",
                    edgecolor="gray",
                    linewidth=2,
                    alpha=1,
                ),
                va="top",
                ha="right",
                alpha=0,
                zorder=10,
            )
            ax.annotate(
                z.values[i],
                xy=(x.values[i], y.values[i]),
                fontsize=10,
                xytext=(-2, -1.5),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="square",
                    facecolor="none",
                    linestyle="--",
                    edgecolor="black",
                    linewidth=1,
                    alpha=1,
                ),
                va="top",
                ha="right",
                alpha=1,
                zorder=10,
            )
