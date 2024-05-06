import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import numpy.lib.recfunctions as rfn
from app.moscat.interfaces.snapshot_quality_measures import SilhouetteCoefficient
from app.moscat.interfaces.temporal_quality_measures import JaccardScore



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class EvolDBSCAN:
    def __init__(self, eps_set, min_pts_set):
        self.eps_set = eps_set
        self.min_pts_set = min_pts_set
        self.sq_metric = SilhouetteCoefficient()
        self.tq_metric = JaccardScore()

    def define_data(self, data_definition):
        self.object_id_col = data_definition["object_id"]
        self.time_col = data_definition["time"]
        self.feature_cols = data_definition["features"]
        self.sq_metric.define_data(data_definition)
        self.tq_metric.define_data(data_definition)

    def _fields_view(self, array, fields):
        arr = array.getfield(
            np.dtype({name: array.dtype.fields[name] for name in fields})
        )
        return arr

    def get_clustering(self, timepoint_data, clustering_param_set):
        data_points = list(
            map(
                lambda tpl: list(tpl),
                self._fields_view(timepoint_data, self.feature_cols),
            )
        )
        cluster_labels = DBSCAN(**clustering_param_set).fit(data_points).labels_
        clustering = rfn.append_fields(
            timepoint_data, "cluster_id", cluster_labels, dtypes="i4"
        )

        return clustering

    def fit(self, timeseries_data):
        times = np.unique(timeseries_data[self.time_col])

        clusterings_over_time = []
        for t in times:
            clusterings = []
            for eps in self.eps_set:
                for min_pts in self.min_pts_set:
                    timepoint_data = timeseries_data[
                        timeseries_data[self.time_col] == t
                    ]
                    clustering_param_set = {"eps": eps, "min_samples": min_pts}
                    clustering = self.get_clustering(
                        timepoint_data, clustering_param_set
                    )
                    row = {
                        "input_parameters": {"eps": eps, "min_pts": min_pts},
                        "clustering": clustering,
                    }
                    clusterings.append(row)
            clusterings_over_time.append(clusterings)

        self.clusterings_over_time = clusterings_over_time

    def select_clustering_over_time(self, weight):
        selected_clusterings = []
        for i, timepoint_clusterings in enumerate(self.clusterings_over_time):

            best_parameter_set = None
            best_oq = -1
            for tp_clustering in timepoint_clusterings:
                sq = self.sq_metric.calculate_score(tp_clustering["clustering"], None)
                if i == 0:
                    hc = 0
                else:
                    prev_parameter_set = AttrDict(selected_clusterings[-1])
                    tq = self.tq_metric.calculate_score(
                        prev_parameter_set, tp_clustering["clustering"], None
                    )
                    hc = 1 - tq
                oq = sq / 2 - weight * hc
                if oq > best_oq:
                    best_parameter_set = tp_clustering
                    best_parameter_set["temporal_quality"] = 1 - hc
                    best_parameter_set["snapshot_quality"] = sq
                    best_oq = oq
            selected_clusterings.append(best_parameter_set)
        return selected_clusterings
