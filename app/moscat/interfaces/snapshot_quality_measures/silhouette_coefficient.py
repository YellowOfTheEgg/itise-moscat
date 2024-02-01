from sklearn import metrics
from app.moscat.interfaces.snapshot_quality_measures.interface import IQualityMeasure
import numpy as np


class SilhouetteCoefficient(IQualityMeasure):

    lower_bound = 0.0
    upper_bound = 2.0

    def define_data(self, data_definition):
        self.col_object_id = data_definition["object_id"]
        self.col_time = data_definition["time"]
        self.col_features = data_definition["features"]

    def _calculate_silhouette_coefficent(self, clustering):

        feature_data = clustering[self.col_features]

        X = list(map(lambda tpl: list(tpl), feature_data))

        labels = clustering["cluster_id"]

        if len(np.unique(labels)) == len(X):
            coeff = 0  # -2.0
        elif len(np.unique(labels)) > 1:
            coeff = 1 + metrics.silhouette_score(X=X, labels=labels)
        else:
            coeff = 0  # -2.0
        return coeff

    def calculate_score(self, clustering, output_parameters):

        return self._calculate_silhouette_coefficent(clustering)
