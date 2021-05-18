import numpy as np
import pandas as pd
from fcmeans import FCM

from ICCluster.ICCluster import ICCluster
from ICCluster.ICValidation import ICValidation


class ICClusterMadsonDias(ICCluster):
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
        model = FCM(n_clusters=num_clusters, max_iter=1000, m=2, error=0.005, random_state=self._seed)
        model.fit(data.to_numpy().T)
        return model

    def _test_model(self, model, data):
        """
        Train model with the number of clusters that has better evaluation
        :param model: Trained model
        :param data: Dataframe with test data
        :return: Tested model
        """
        super()._test_model(model, data)
        model.predict(data.to_numpy().T)

        # Evaluation
        validation = ICValidation(model.partition_coefficient)
        return validation

    def get_clusters(self):
        """
        Get clusters created by the model
        :return: List of clusters with respective ICs
        """
        super().get_clusters()
        cluster_membership = np.argmax(self._model.u, axis=1)
        return self._get_cluster_members(cluster_membership)

    def get_fuzzy_partition(self):
        return pd.DataFrame(self._model.u.T, columns=self._icimport.get_code_dict().keys())

    def get_model_centers(self):
        return self._model.centers
