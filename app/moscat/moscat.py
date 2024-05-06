import numpy as np


class Moscat:
    def __init__(
        self,
        clustering_method,
        snapshot_quality_measure,
        temporal_quality_measure,
        pareto_selection_strategy,
        log_intermediate_steps=False,
    ):
        self.clustering_method = clustering_method

        self.snapshot_quality_measure = snapshot_quality_measure
        self.temporal_quality_measure = temporal_quality_measure
        self.pareto_selection_strategy = pareto_selection_strategy
        self.log_intermediate_steps = log_intermediate_steps

        self.optimal_parameters = []
        self.intermediate_steps = []

    def define_data(self, data_definition):
        self.col_object_id = data_definition["object_id"]
        self.col_time = data_definition["time"]
        self.col_features = data_definition["features"]
        self.snapshot_quality_measure.define_data(data_definition)
        self.temporal_quality_measure.define_data(data_definition)
        self.clustering_method.define_data(data_definition)

    def calculate_init_params(self, data):
        init_parameters = []
        for clustering_params in self.clustering_method.input_parameters:

            clustering, output_parameters = self.clustering_method.do_clustering(
                data, clustering_params, past_clustering_info=None
            )

            snapshot_quality = self.snapshot_quality_measure.calculate_score(
                clustering, output_parameters
            )
            if snapshot_quality is not None:
                init_parameters.append(
                    StepParameterSet(
                        snapshot_quality,
                        0,
                        clustering_params,
                        clustering,
                        output_parameters,
                    )
                )

        return init_parameters

    def extract_optimum_from_init_params(self, init_params):
        optimal_params = max(init_params, key=lambda row: row.snapshot_quality)
        return optimal_params

    def calculate_step_parameters(self, past_clustering_info, data_2):
        result = []

        for clustering_params in self.clustering_method.input_parameters:
            current_clustering, output_parameters = self.clustering_method.do_clustering(
                data_2, clustering_params, past_clustering_info=past_clustering_info
            )

            snapshot_quality = self.snapshot_quality_measure.calculate_score(
                current_clustering, output_parameters
            )
            temporal_quality = self.temporal_quality_measure.calculate_score(
                past_clustering_info, current_clustering, output_parameters
            )

            result.append(
                StepParameterSet(
                    snapshot_quality,
                    temporal_quality,
                    clustering_params,
                    current_clustering,
                    output_parameters,
                )
            )
        return result

    def extract_pareto_front(self, possible_step_parameters):
        front = []
        for set_b in possible_step_parameters:
            dominated = False
            for set_a in possible_step_parameters:
                if (
                    set_a.snapshot_quality == set_b.snapshot_quality
                    and set_a.temporal_quality > set_b.temporal_quality
                ):
                    dominated = True
                    break
                elif (
                    set_a.snapshot_quality > set_b.snapshot_quality
                    and set_a.temporal_quality == set_b.temporal_quality
                ):
                    dominated = True
                    break
                elif (
                    set_a.snapshot_quality > set_b.snapshot_quality
                    and set_a.temporal_quality > set_b.temporal_quality
                ):
                    dominated = True
                    break

            if not dominated:
                front.append(set_b)
        return front

    def extract_optimum_from_pateto_front(self, pareto_front):

        return self.pareto_selection_strategy.extract_optimal_score(pareto_front)

    def remove_redundant_step_parameters(self, step_parameter_sets):
        unique_sets = []
        for step_parameter_set in step_parameter_sets:
            if step_parameter_set not in unique_sets:
                unique_sets.append(step_parameter_set)

        # return unique_sets
        with_sqs = [x for x in unique_sets if x.snapshot_quality > 0]
        if len(with_sqs) == 0:
            return unique_sets
        else:
            return with_sqs

    def remove_zero_sqs(self, step_parameter_sets):
        return [x for x in step_parameter_sets if x.snapshot_quality > 0]

    def calculate_optimal_parameters(self, data):
        times = np.unique(data[self.col_time])
        init_timepoint_data = data[data[self.col_time] == times[0]]

        init_parameter_sets = self.calculate_init_params(init_timepoint_data)
        init_parameter_sets = self.remove_redundant_step_parameters(init_parameter_sets)
        optimal_init_parameter_set = self.extract_optimum_from_init_params(
            init_parameter_sets
        )

        self.optimal_parameters.append(optimal_init_parameter_set)
        if self.log_intermediate_steps:
            self.intermediate_steps.append(
                {
                    "available_step_parameters": init_parameter_sets,
                    "pareto_front": [optimal_init_parameter_set],
                }
            )

        for time in times[1:]:
            timepoint_data = data[data[self.col_time] == time]
            available_step_parameters = self.calculate_step_parameters(
                self.optimal_parameters[-1], timepoint_data
            )
            available_step_parameters = self.remove_redundant_step_parameters(
                available_step_parameters
            )
            pareto_front = self.extract_pareto_front(available_step_parameters)

            if self.log_intermediate_steps:
                self.intermediate_steps.append(
                    {
                        "available_step_parameters": available_step_parameters,
                        "pareto_front": pareto_front,
                    }
                )

            optimal_step_parameter_set = self.extract_optimum_from_pateto_front(
                pareto_front
            )

            self.optimal_parameters.append(optimal_step_parameter_set)


class StepParameterSet:
    def __init__(
        self,
        snapshot_quality,
        temporal_quality,
        input_parameters,
        clustering,
        output_parameters=[],
    ):
        self.temporal_quality = temporal_quality
        self.snapshot_quality = snapshot_quality
        self.input_parameters = input_parameters
        self.output_parameters = output_parameters
        self.clustering = clustering

    def __eq__(self, step_parameter_set):
        return (
            self.snapshot_quality == step_parameter_set.snapshot_quality
            and self.temporal_quality == step_parameter_set.temporal_quality
        )

    def __dict__(self):
        return {
            "temporal_quality": self.temporal_quality,
            "snapshot_quality": self.snapshot_quality,
            "input_parameters": self.input_parameters,
            "output_parameters": self.output_parameters,
            "clustering": self.clustering,
        }

    def __str__(self):
        return str(
            {
                "temporal_quality": self.temporal_quality,
                "snapshot_quality": self.snapshot_quality,
                "input_parameters": self.input_parameters,
                "output_parameters": self.output_parameters,
                "clustering": "[too long to show]",
            }
        )

    def __repr__(self):
        return str(
            {
                "temporal_quality": self.temporal_quality,
                "snapshot_quality": self.snapshot_quality,
                "input_parameters": self.input_parameters,
                "output_parameters": self.output_parameters,
                "clustering": "[too long to show]",
            }
        )
