from ICCluster.ICValidation import ICValidation
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class ICCluster(ABC):
    """
    Represents the Clustering model of the Integrated Circuits analysis
    """

    def __init__(self, icimport=None, num_clusters=None):
        self._icimport = icimport
        self._seed = 200
        self._num_folds = 10
        self._num_clusters = num_clusters
        self._min_clusters = 2
        self._max_clusters = 10
        self._cv_cluster_search = None

        if icimport is not None:
            self._model = self._generate_model(icimport, self._num_clusters)
            self._cv = self._cross_validation(icimport, self._num_clusters)

    def __str__(self):
        cv_dict = self._cv.get_validity_dict()
        idx = 0
        text = "Country: " + self._icimport.get_country() + '\n'
        text += "\t-> Code:   " + self._icimport.get_code_standard() + '\n'
        text += "\t-> Source: " + self._icimport.get_source() + '\n'
        text += "\t-> Validity:\n"
        for valid in cv_dict:
            text += "\t\t" + valid + ": " + str(cv_dict[valid][0]) + " +- " + str(cv_dict[valid][1]) + '\n'
        text += "\t-> Fuzzy partition:\n"
        text += self.get_fuzzy_partition().to_string()
        return text

    def _generate_model(self, icimport, num_clusters=None):
        """
        Generate a clustering model
        :param icimport: ICImport object
        :param num_clusters: Number of clusters
        :return: Trained model
        """
        if num_clusters is None:
            return self._train_best_model(icimport, self._min_clusters, self._max_clusters)
        else:
            return self._train_model(icimport.get_data(), num_clusters)

    def _cross_validation(self, icimport, num_clusters):
        """
        Perform cross validation of dataset with defined number of clusters
        :param icimport: ICImport object
        :param num_clusters: Number of clusters to evaluate
        :return: Results dictionary of the cross_validation iterations
        """
        num_samples = icimport.get_num_ics() // self._num_folds
        orig_df = icimport.get_data().copy()

        # Creating K folds
        folds = []
        for _ in range(self._num_folds):
            fold = orig_df.sample(n=num_samples, random_state=self._seed, axis=1)
            orig_df.drop(fold.columns, axis=1, inplace=True)
            folds += [fold]
        for idx in range(len(orig_df.columns)):
            folds[idx % self._num_folds][orig_df.columns[idx]] = orig_df[orig_df.columns[idx]]

        # Running cross validations
        results = []
        for test in folds:

            # Train and test
            train = pd.concat([fold for fold in folds if fold is not test], axis=1).sort_index()
            model = self._train_model(train, num_clusters)
            result = self._test_model(model, test)
            results += [result]

        # Resulting evaluation
        result = ICValidation()
        result.mean_validity(results)
        return result

    def _train_best_model(self, icimport, min_clusters, max_clusters):
        """
        Train model with the number of clusters that has better evaluation
        :param icimport: ICImport object
        :param min_clusters: Minimum number of clusters to try
        :param max_clusters: Maximum number of clusters to try
        :return: Trained model
        """
        # Get evaluation with different numbers of clusters
        results = []
        for num_clusters in range(min_clusters, max_clusters + 1):
            results += [self._cross_validation(icimport, num_clusters)]
        self._cv_cluster_search = results

        # Return best model
        evals = [r.get_eval() for r in results]
        best_num_clusters = min_clusters + evals.index(max(evals))
        return self._train_model(icimport.get_data(), best_num_clusters)

    def _get_cluster_members(self, cluster_membership):
        clusters = []
        for idx in range(self._num_clusters):
            mask = [cm != idx for cm in cluster_membership]
            ic_masked = np.ma.masked_array(self._icimport.get_ics(), mask=mask)
            clusters += [np.ma.getdata(ic_masked[ic_masked.mask == False])]
        return clusters

    @abstractmethod
    def _train_model(self, data, num_clusters):
        self._num_clusters = num_clusters
        pass

    @abstractmethod
    def _test_model(self, model, data):
        pass

    def roc(self):
        # TODO: May use ROC curve to validate model
        # - plot "true positives" versus "false positives"
        # - Percentage TP = 100 * TP / (TP + FN), where TP = True Positive,  FN = False Negative
        # - Percentage FP = 100 * FP / (FP + TN), where FP = False Positive, TN = True Negative
        return None

    def get_model(self):
        return self._model

    def get_cross_validation(self):
        return self._cv

    def get_cluster_search(self):
        """
        Get cross-validation results of search for optimal cluster number
        :return: List of ICValidation and list of cluster numbers
        """
        return self._cv_cluster_search, list(range(self._min_clusters, self._max_clusters+1))

    @abstractmethod
    def get_clusters(self):
        pass

    @abstractmethod
    def get_fuzzy_partition(self):
        pass

    @abstractmethod
    def get_model_centers(self):
        pass


