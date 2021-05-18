import numpy as np
import pandas as pd
import skfuzzy as fuzz

from ICCluster.ICCluster import ICCluster
from ICCluster.ICValidation import ICValidation


class ICClusterSciKit(ICCluster):
    """
    ICCluster implementation using SciKit c-means functions
    """

    def __init__(self, df=None):
        super().__init__(df)

    def _train_model(self, data, num_clusters):
        """
        Train model with the number of clusters that has better evaluation
        :param data: Dataframe with train data
        :param num_clusters: Number of clusters to use
        :return: Trained model
        """
        super()._train_model(data, num_clusters)
        data_array = data.to_numpy()
        return fuzz.cmeans(data=data_array, c=num_clusters, m=2, error=0.005, maxiter=1000)

    def _test_model(self, model, data):
        """
        Train model with the number of clusters that has better evaluation
        :param model: Trained model
        :param data: Dataframe with test data
        :return: Tested model
        """
        super()._test_model(model, data)
        data_array = data.to_numpy()
        result = fuzz.cmeans_predict(test_data=data_array, cntr_trained=model[0], m=2, error=0.005, maxiter=1000)

        # Evaluation
        validation = ICValidation(result[5])
        return validation

    def get_clusters(self):
        """
        Get clusters created by the model
        :return: List of clusters with respective ICs
        """
        super().get_clusters()
        cluster_membership = np.argmax(self._model[1], axis=0)
        return self._get_cluster_members(cluster_membership)

    def get_fuzzy_partition(self):
        return pd.DataFrame(self._model[1], columns=self._icimport.get_code_dict().keys())

    def get_model_centers(self):
        return self._model[0]
