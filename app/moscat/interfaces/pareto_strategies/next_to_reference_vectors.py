import math
from app.moscat.interfaces.pareto_strategies.interface import IParetoStrategy


class NextToReferenceVectors(IParetoStrategy):
    def __init__(self, max_snapshot_quality, max_temporal_quality, weight):
        self.max_snapshot_quality = max_snapshot_quality
        self.max_temporal_quality = max_temporal_quality
        self.weight = weight

    def is_point_over_hypothenuse(self, step_parameter_set):
        max_x = self.max_snapshot_quality
        max_y = self.max_temporal_quality
        current_x = step_parameter_set.snapshot_quality
        current_y = step_parameter_set.temporal_quality
        hypothenuse_y = max_y - (max_y / max_x) * current_x
        return hypothenuse_y < current_y

    def calculate_distance_over_hypothenuse(self, step_parameter_set):
        x_max = self.max_snapshot_quality
        y_max = self.max_temporal_quality
        x_current = step_parameter_set.snapshot_quality
        y_current = step_parameter_set.temporal_quality
        w = self.weight
        nominator = abs(
            x_current * y_max * w
            + y_current * x_max * w
            - y_current * x_max
            - 2 * y_max * x_max * w
            + y_max * x_max
        )

        denominator = math.sqrt(
            (x_max ** 2) * w ** 2
            - 2 * (x_max ** 2) * w
            + x_max ** 2
            + (y_max ** 2) * (w ** 2)
        )
        return nominator / denominator

    def calculate_distance_under_hypothenuse(self, step_parameter_set):
        x_max = self.max_snapshot_quality
        y_max = self.max_temporal_quality
        x_current = step_parameter_set.snapshot_quality
        y_current = step_parameter_set.temporal_quality
        w = self.weight
        nominator = abs(
            -x_current * y_max * w + x_current * y_max - y_current * x_max * w
        )
        denominator = math.sqrt(
            (x_max ** 2) * (w ** 2)
            + (y_max ** 2) * (w ** 2)
            - 2 * (y_max ** 2) * w
            + (y_max ** 2)
        )

        return nominator / denominator

    def calculate_distance(self, step_parameter_set):
        over_hypothenuse = self.is_point_over_hypothenuse(step_parameter_set)
        distance = None
        if over_hypothenuse:
            distance = self.calculate_distance_over_hypothenuse(step_parameter_set)
        else:
            distance = self.calculate_distance_under_hypothenuse(step_parameter_set)
        return distance

    def extract_optimal_score(self, pareto_front):
        step_parameter_set_distances = []
        for step_parameter_set in pareto_front:
            distance = self.calculate_distance(step_parameter_set)
            step_parameter_set_distances.append((step_parameter_set, distance))

        optimal_params = min(step_parameter_set_distances, key=lambda p: p[1])
        return optimal_params[0]

    def extract_reference_vectors(self):
        vector_0 = [
            [0, 0],
            [
                self.weight * self.max_snapshot_quality,
                self.max_temporal_quality - self.max_temporal_quality * self.weight,
            ],
        ]

        vector_1 = [
            [
                self.weight * self.max_snapshot_quality,
                self.max_temporal_quality - self.max_temporal_quality * self.weight,
            ],
            [self.max_snapshot_quality, self.max_temporal_quality],
        ]
        return [vector_0, vector_1]

    def extract_orthogonal_line(self):
        return [[0, self.max_temporal_quality], [self.max_snapshot_quality, 0]]
