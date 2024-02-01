import numpy.lib.recfunctions as rfn
import numpy as np
from sklearn.cluster import DBSCAN as SLDBSCAN
from app.moscat.interfaces.clustering_methods.interface import IClusteringMethod


class DBSCAN(IClusteringMethod):
    def __init__(self, eps_range=[0, 1, 0.1], minpts_range=[2, 3, 1]):
        self.eps_range = eps_range
        self.minpts_range = minpts_range
        self.input_parameters = []
        self.generate_input_parameters()

    def _fields_view(self, array, fields):
        arr = array.getfield(
            np.dtype({name: array.dtype.fields[name] for name in fields})
        )
        return arr

    def define_data(self, data_definition):
        self.data_definition = data_definition

    def do_clustering(self, data, clustering_params, past_clustering_info=None):

        feature_data = list(
            map(
                lambda tpl: list(tpl),
                self._fields_view(data, self.data_definition["features"]),
            )
        )
        cluster_labels = SLDBSCAN(**clustering_params).fit(feature_data).labels_
        clustering = rfn.append_fields(data, "cluster_id", cluster_labels, dtypes="i4")

        #        current_negative_cluster_id = -1
        #        cluster_ids = []
        #        for cluster_id in clustering["cluster_id"]:
        #            if cluster_id < 0:
        #                cluster_ids.append(current_negative_cluster_id)
        #                current_negative_cluster_id -= 1
        #            else:
        #                cluster_ids.append(cluster_id)

        #        clustering["cluster_id"] = cluster_ids
        return clustering, None

    def generate_input_parameters(self):
        minpts_list = list(np.arange(*self.minpts_range))
        eps_list = list(map(lambda x: round(x, 2), np.arange(*self.eps_range)))
        parameters_to_prove = []
        for minpts in minpts_list:
            for eps in eps_list:
                parameters_to_prove.append({"eps": eps, "min_samples": minpts})
        self.input_parameters = parameters_to_prove
