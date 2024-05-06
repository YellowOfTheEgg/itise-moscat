import numpy.lib.recfunctions as rfn
from app.moscat.interfaces.clustering_methods.interface import IClusteringMethod
import numpy as np
from app.utilities.custom_kmeans.incremental_kmeans import IncrementalKmeans


class IIncrementalKmeans(IClusteringMethod):
    def __init__(self, k_set, max_iter_set, init_point_index, halflife=1):
        self.input_parameters = []
        self.k_set = k_set
        self.halflife = halflife
        self.max_iter_set = max_iter_set
        self.init_point_index = init_point_index

        self.generate_input_parameters()

    def define_data(self, data_definition):
        self.data_definition = data_definition

    def _fields_view(self, array, fields):
        return array.getfield(
            np.dtype({name: array.dtype.fields[name] for name in fields})
        )

    def do_clustering(self, data, input_parameters, past_clustering_info=None):
        if past_clustering_info is None:
            initial_centroids = []
        else:
            initial_centroids = np.copy(
                past_clustering_info.output_parameters["centroids"]
            )

        clustering = IncrementalKmeans(
            **input_parameters,
            halflife=self.halflife,
            initial_centroids=initial_centroids
        )
        clustering.define_data(self.data_definition)
        clustering.fit(data)
        clustering_result = clustering.clustering_result
        output_parameters = clustering.output_parameters

        return clustering_result, output_parameters

    def generate_input_parameters(self):
        for k in self.k_set:
            for max_iter in self.max_iter_set:
                self.input_parameters.append(
                    {
                        "n_clusters": k,
                        "max_iter": max_iter,
                        "init_point_index": self.init_point_index,
                    }
                )
