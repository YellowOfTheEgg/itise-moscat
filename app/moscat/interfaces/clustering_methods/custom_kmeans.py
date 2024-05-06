import numpy.lib.recfunctions as rfn
from app.moscat.interfaces.clustering_methods.interface import IClusteringMethod
import numpy as np
from app.utilities.custom_kmeans.custom_kmeans import Kmeans


class CUSTOMKMEANS(IClusteringMethod):
    def __init__(self, k_set, max_iter_set, init_point_index_set, step_size=1):
        self.clustering_parameters_set = []
        self.k_set = k_set
        self.step_size = step_size
        self.max_iter_set = max_iter_set
        self.init_point_index_set = init_point_index_set
        self.generate_clustering_parameters_set()

    def define_data(self, data_definition):
        self.data_definition = data_definition

    def _fields_view(self, array, fields):
        return array.getfield(
            np.dtype({name: array.dtype.fields[name] for name in fields})
        )

    def do_clustering(self, data, clustering_params):
        clustering = Kmeans(**clustering_params, step_size=self.step_size)
        clustering.define_data(self.data_definition)
        clustering.fit(data)
        clustering = clustering.clustering_result
        # return structured array with fields: cluster_id, time, feature_1, feature_2, cluster_id
        return clustering

    def generate_clustering_parameters_set(self):
        for k in self.k_set:
            for max_iter in self.max_iter_set:
                for init_point_index in self.init_point_index_set:
                    self.clustering_parameters_set.append(
                        {
                            "n_clusters": k,
                            "max_iter": max_iter,
                            "init_point_index": init_point_index,
                        }
                    )
