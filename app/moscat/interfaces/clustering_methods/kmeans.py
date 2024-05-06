import numpy.lib.recfunctions as rfn
from app.moscat.interfaces.clustering_methods.interface import IClusteringMethod
import numpy as np
from sklearn.cluster import KMeans as SLKMeans


class KMEANS(IClusteringMethod):
    def __init__(self, k_range, init_times=1):
        self.clustering_parameters_set = []
        self.k_range = k_range
        self.init_times = init_times
        self.generate_clustering_parameters_set()

    def _fields_view(self, array, fields):
        return array.getfield(
            np.dtype({name: array.dtype.fields[name] for name in fields})
        )

    def do_clustering(self, data, clustering_params, feature_names):
        feature_data = list(
            map(lambda tpl: list(tpl), self._fields_view(data, feature_names))
        )  # is a cloud of points, each point is a 2d list
        clustering = SLKMeans(**clustering_params).fit(feature_data)
        cluster_labels = clustering.labels_
        clustering = rfn.append_fields(data, "cluster_id", cluster_labels, dtypes="i4")

        return clustering

    def generate_clustering_parameters_set(self):
        k_set = list(np.arange(self.k_range[0], self.k_range[1], 1)) * self.init_times
        for k in k_set:
            self.clustering_parameters_set.append(
                {"n_clusters": k, "init": "random", "n_init": 10}
            )
