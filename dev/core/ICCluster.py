from fcmeans import FCM
import skfuzzy as fuzz
from abc import ABC, abstractmethod


class ICCluster(ABC):
    """
    Represents the Clustering model of the Integrated Circuits analysis
    """

    def __init__(self, df=None):
        self._df = df
        self._seed = 200
        self._num_folds = 10
        self._num_clusters = None

        self._model = self.generate_model(df)
        # TODO: Create a class for Cross Validation maybe?
        self._cv = self.cross_validation()

    def generate_model(self, df=None, num_clusters=None):
        """
        Generate a clustering model
        :param df: Data to train model
        :param num_clusters: Number of clusters
        :return: Trained model
        """
        if df is None:
            return None
        else:
            if num_clusters is None:
                return self._get_best_model(self._df, 2, 5)
            else:
                self._num_clusters = num_clusters
                return self._train_model(self._df, num_clusters)

    def cross_validation(self):
        """
        Perform cross validation of dataset with defined number of clusters
        :return: Results of the cross_validation iterations
        """
        if self._model is None:
            return None

        num_folds = self._num_folds

        num_samples = len(self._df) // num_folds
        orig_df = self._df.copy()

        # Creating K folds
        folds = []
        for _ in range(num_folds):
            fold = orig_df.sample(n=num_samples, random_state=self._seed)
            orig_df.drop(fold.index)
            folds += [fold]

        # Running cross validations
        results = []
        for test in folds:
            train = [fold for fold in folds if fold is not test]
            model = self._train_model(train, self._num_clusters)
            results += [self._test_model(model, test)]

        return results

    def roc(self):
        # TODO: May use ROC curve to validate model
        # - plot "true positives" versus "false positives"
        # - Percentage TP = 100 * TP / (TP + FN), where TP = True Positive,  FN = False Negative
        # - Percentage FP = 100 * FP / (FP + TN), where FP = False Positive, TN = True Negative
        return None

    def _get_best_model(self, df=None, min_clusters=2, max_clusters=5):
        if df is None:
            return None

        # Get models with different numbers of clusters
        models = []
        for num_clusters in range(min_clusters, max_clusters + 1):
            models += [self._train_model(df, num_clusters)]

        # TODO: Best model logic (must use FPC and indexes of reference articles)
        idx = 0
        self._num_clusters = min_clusters + idx
        return models[idx]

    @abstractmethod
    def _train_model(self, train, num_clusters):
        return None

    @abstractmethod
    def _test_model(self, model, test):
        return None


class ICClusterSciKit(ICCluster):

    def __init__(self, df=None):
        super().__init__(df)

    def _train_model(self, data, num_clusters):
        return fuzz.cmeans(data=data, c=num_clusters, m=2, error=0.005, maxiter=1000)

    def _test_model(self, model, data):
        return fuzz.cmeans_predict(test_data=data, cntr_trained=model[0], m=2, error=0.005, maxiter=1000)


class ICClusterDiasMLD(ICCluster):

    def __init__(self, df=None):
        super().__init__(df)

    def _train_model(self, data, num_clusters):
        data_array = data.to_numpy()
        fcm = FCM(n_clusters=num_clusters, max_iter=1000, m=2, error=0.005, random_state=self._seed)
        fcm.fit(data_array)
        return fcm

    def _test_model(self, model, data):
        data_array = data.to_numpy()
        return model.predict(data_array)
