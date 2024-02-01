from abc import ABC, abstractmethod


class IClusteringMethod(ABC):
    clustering_parameter_set: list

    @abstractmethod
    def do_clustering(self, data, clustering_params, feature_names):
        pass

    @abstractmethod
    def generate_input_parameters(self):
        pass
