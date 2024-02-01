from abc import ABC, abstractmethod


class ITemporalQualityMeasure(ABC):
    lower_bound: float
    upper_bound: float

    @abstractmethod
    def calculate_score(self, clustering_1, clustering_2):
        pass

    @abstractmethod
    def define_data(self, data_definition):
        pass
