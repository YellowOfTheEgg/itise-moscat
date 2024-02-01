from abc import ABC, abstractmethod


class IQualityMeasure(ABC):
    lower_bound: float
    upper_bound: float

    @abstractmethod
    def calculate_score(self, clustering):
        pass

    @abstractmethod
    def define_data(self, data_definition):
        pass
