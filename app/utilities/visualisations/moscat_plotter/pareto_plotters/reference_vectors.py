import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from .interface import IParetoPlotter
import math


class VisualElements:
    def reference_vector(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        handle1 = plt.arrow(
            x1,
            y1,
            dx,
            dy,
            linestyle="-",
            color="#007fff",
            zorder=2,
            head_width=0.02,
            alpha=0.8,
            length_includes_head=True,
        )
        handle2 = plt.arrow(
            x1,
            y1,
            dx,
            dy,
            linestyle="-",
            color="#eaeaf2",
            zorder=1,
            head_width=0.02,
            alpha=1,
        )
        return ((handle1, handle2), "Reference vectors")

    def bending_line(self, x1, y1, x2, y2):
        handle = plt.plot([x1, x2], [y1, y2], linestyle="--", color="grey", alpha=0.5)
        return handle

    def pareto_element(
        self, x, y, clustering_params, color, zorder_offset, point_size=200
    ):
        label = "(" + ", ".join(str(v) for v in clustering_params.values()) + ")"
        handle_1 = plt.scatter(
            x,
            y,
            label="part of pareto front",
            marker="o",
            s=point_size,
            color=(0, 0, 0, 0),
            edgecolor=(0, 0, 0, 1),
            linewidth=1,
            #  alpha=0.5,
            zorder=zorder_offset + 6,
        )
        handle_2 = plt.scatter(
            x,
            y,
            label=label,
            marker="o",
            color=color,
            s=point_size,
            alpha=0.5,
            zorder=zorder_offset + 5,
        )
        handle_3 = plt.scatter(
            x, y, marker="o", s=point_size, c="#eaeaf2", zorder=zorder_offset + 4
        )
        return ((handle_3, handle_2, handle_1), label)

    def selected_element(
        self, x, y, clustering_params, color, zorder_offset, point_size=200
    ):
        label = "(" + ", ".join(str(v) for v in clustering_params.values()) + ")"

        handle_1 = plt.scatter(
            x, y, marker="o", s=point_size, c="#eaeaf2", zorder=zorder_offset + 4
        )

        handle_2 = plt.scatter(
            x,
            y,
            label=label,
            marker="o",
            color=color,
            s=point_size,
            alpha=0.5,
            zorder=zorder_offset + 5,
        )
        handle_3 = plt.scatter(
            x,
            y,
            label="part of pareto front",
            marker="o",
            s=point_size,
            color=(0, 0, 0, 0),
            edgecolor=(0, 0, 0, 1),
            linewidth=1,
            zorder=zorder_offset + 6,
        )

        handle_4 = plt.scatter(
            x,
            y,
            marker="o",
            s=point_size - 120,
            edgecolor=(0, 0, 0, 1),
            color=(0, 0, 0, 0),
            linewidth=1,
            zorder=zorder_offset + 7,
        )
        return ((handle_1, handle_2, handle_3, handle_4), label)

    def scatter_element(
        self, x, y, clustering_params, color, zorder_offset, point_size=200
    ):
        label = "(" + ", ".join(str(v) for v in clustering_params.values()) + ")"
        handle_1 = plt.scatter(
            x,
            y,
            label=label,
            marker="o",
            color=color,
            s=point_size,
            alpha=0.5,
            zorder=zorder_offset + 2,
        )
        handle_2 = plt.scatter(
            x, y, marker="o", s=point_size, c="#eaeaf2", zorder=zorder_offset + 1
        )

        return ((handle_2, handle_1), label)

    def pareto_front_line(self, x, y, zorder_offset):

        plt.plot(
            x, y, linestyle="-", color="black", alpha=0.7, zorder=zorder_offset + 3
        )


class ReferenceVectors(IParetoPlotter):
    def __init__(
        self,
        clevalot,
        guide_line_1=None,
        guide_line_2=None,
        guide_line_3=None,
        clustering_params_legend_mapping={},
    ):
        self.clevalot = clevalot
        self.guide_line_1 = guide_line_1
        self.guide_line_2 = guide_line_2
        self.guide_line_3 = guide_line_3
        self.clustering_params_legend_mapping = clustering_params_legend_mapping
        self.vis_elements = VisualElements()
        sns.set(style="darkgrid", font="Arial", font_scale=1.4)
        plt.figure(figsize=(10, 6))

    def add_guide_lines(self):
        if self.guide_line_3 is not None:
            self.vis_elements.bending_line(
                x1=self.guide_line_3[0][0],
                y1=self.guide_line_3[0][1],
                x2=self.guide_line_3[1][0],
                y2=self.guide_line_3[1][1],
            )

        if self.guide_line_1 is not None:
            self.vis_elements.reference_vector(
                x1=self.guide_line_1[0][0],
                y1=self.guide_line_1[0][1],
                x2=self.guide_line_1[1][0],
                y2=self.guide_line_1[1][1],
            )

        if self.guide_line_2 is not None:
            self.vis_elements.reference_vector(
                x1=self.guide_line_2[0][0],
                y1=self.guide_line_2[0][1],
                x2=self.guide_line_2[1][0],
                y2=self.guide_line_2[1][1],
            )

    def add_plot_settings(self):
        plt.ylabel("Temporal Quality")
        plt.xlabel("Snapshot Quality")
        plt.xlim(
            -0.1, self.clevalot.pareto_selection_strategy.max_snapshot_quality + 0.1
        )
        plt.ylim(
            -0.1, self.clevalot.pareto_selection_strategy.max_temporal_quality + 0.1
        )

    def add_scores(self, available_parameters, pareto_front, optimal_parameter_set):

        point_size = 200
        colors = sns.color_palette()
        zorder_offset = 2

        pareto_coordinates = list(
            map(lambda p: [p.snapshot_quality, p.temporal_quality], pareto_front)
        )
        sorted_pareto_coordinates = sorted(pareto_coordinates, key=lambda x: x[0])

        pareto_front_x = list(map(lambda p: p[0], sorted_pareto_coordinates))
        pareto_front_y = list(map(lambda p: p[1], sorted_pareto_coordinates))

        self.vis_elements.pareto_front_line(
            pareto_front_x, pareto_front_y, zorder_offset=zorder_offset
        )
        scatter_handles = []
        scatter_labels = []
        for idx, parameter_set in enumerate(available_parameters):
            color_id = idx
            if idx >= len(colors):
                color_id = idx % len(colors)
            color = colors[color_id] + (0.5,)
            if parameter_set == optimal_parameter_set:
                handle, label = self.vis_elements.selected_element(
                    x=parameter_set.snapshot_quality,
                    y=parameter_set.temporal_quality,
                    clustering_params=self.remap_score_labels(
                        parameter_set.input_parameters
                    ),
                    color=color,
                    zorder_offset=zorder_offset,
                    point_size=point_size,
                )
            elif parameter_set in pareto_front:
                handle, label = self.vis_elements.pareto_element(
                    x=parameter_set.snapshot_quality,
                    y=parameter_set.temporal_quality,
                    clustering_params=self.remap_score_labels(
                        parameter_set.input_parameters
                    ),
                    color=color,
                    zorder_offset=zorder_offset,
                    point_size=point_size,
                )
            else:
                handle, label = self.vis_elements.scatter_element(
                    x=parameter_set.snapshot_quality,
                    y=parameter_set.temporal_quality,
                    clustering_params=self.remap_score_labels(
                        parameter_set.input_parameters
                    ),
                    color=color,
                    zorder_offset=zorder_offset,
                    point_size=point_size,
                )
            scatter_handles.append(handle)
            scatter_labels.append(label)
        return (scatter_handles, scatter_labels)

    def add_scatter_legend(
        self, intermediate_step, scatter_handles, scatter_labels, denominator=5
    ):

        clustering_parameter_keys = intermediate_step["available_step_parameters"][
            0
        ].input_parameters.keys()
        legend_title = (
            f"Clustering parameters: ({', '.join(list(clustering_parameter_keys))})"
        )
        clustering_parameter_keys = map(
            lambda k: self.clustering_params_legend_mapping.get(k, None),
            clustering_parameter_keys,
        )
        clustering_parameter_keys = list(
            filter(lambda k: k is not None, clustering_parameter_keys)
        )
        for idx, k in enumerate(clustering_parameter_keys):
            if idx < len(clustering_parameter_keys) - 1:
                legend_title += f"{k}, "
            else:
                legend_title += f"{k}):"

        number_of_rows = math.ceil(len(scatter_handles) / denominator)

        height = 0.065 + number_of_rows * 0.033
        y_pos = 1.3 + number_of_rows * 0.033

        legend = plt.legend(
            handles=scatter_handles,
            labels=scatter_labels,
            loc="upper left",
            ncol=5,
            title=legend_title,
            bbox_to_anchor=(0, y_pos, 1.0, height),
            mode="expand",
            borderaxespad=0.0,
            fancybox=False,
        )
        legend._legend_box.align = "left"
        return legend

    def add_general_legend(self):
        from matplotlib.lines import Line2D

        title = "General information"
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=(0, 0, 0, 1),
                markerfacecolor="#eaeaf2",
                markeredgecolor="black",
                markersize=14,
            ),
            Line2D([0], [0], linestyle="--", color=(0, 0, 0, 0.5)),
            (
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=(0, 0, 0, 0),
                    markeredgecolor="black",
                    label="Element of pareto front",
                    markersize=9,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=(0, 0, 0, 0),
                    markeredgecolor="black",
                    label="Element of pareto front",
                    markersize=14,
                ),
            ),
            Line2D([0], [0], marker=">", color="#007fff", alpha=0.8),
        ]

        legend = plt.legend(
            handles=legend_elements,
            labels=[
                "Element of pareto front",
                "Bending line",
                "Selected element from pareto front",
                "Reference vectors",
            ],
            loc="upper left",
            ncol=2,
            title=title,
            bbox_to_anchor=(0, 1.2, 1.0, 0.05),
            mode="expand",
            borderaxespad=0.0,
            fancybox=False,
        )
        legend._legend_box.align = "left"
        return legend

    def remap_score_labels(self, clustering_parameters):

        if len(self.clustering_params_legend_mapping) > 0:
            remaped_score_labels = {}
            for k, v in clustering_parameters.items():
                if k in self.clustering_params_legend_mapping.keys():
                    new_key = self.clustering_params_legend_mapping[k]
                    remaped_score_labels[new_key] = v
        else:
            remaped_score_labels = clustering_parameters
        return remaped_score_labels

    def create_plot(self, img_path):

        legend_denominator = 5
        for step_idx, step in enumerate(
            tqdm(self.clevalot.intermediate_steps, desc="Creating plots")
        ):

            self.add_guide_lines()
            handles, labels = self.add_scores(
                step["available_step_parameters"],
                step["pareto_front"],
                self.clevalot.optimal_parameters[step_idx],
            )

            self.add_plot_settings()
            legend_2 = self.add_general_legend()
            legend_1 = self.add_scatter_legend(
                self.clevalot.intermediate_steps[0], handles, labels, legend_denominator
            )

            folder_exists = os.path.exists(img_path)
            if not folder_exists:
                os.makedirs(img_path)
            plt.gca().add_artist(legend_2)
            plt.savefig(
                f"{img_path}/step_{step_idx}.png",
                bbox_extra_artists=(legend_2, legend_1),
                bbox_inches="tight",
            )

            plt.clf()
