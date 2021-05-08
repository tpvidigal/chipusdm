import pandas as pd
from sklearn.model_selection import cross_val_score
import skfuzzy as fuzz


class ICCluster:
    """
    Represents the Clustering model of the Integrated Circuits analysis
    """

    def __init__(self, df=None):
        self._df = df
        self._seed = 200
        self._num_folds = 10

        self._model = self.generate_model(df)
        # TODO: Crete a class for cross validation
        self._cv = self._cross_validation(self._model, df, self._num_folds)

    def generate_model(self, df=None, num_clusters=None):
        if df is None:
            return None
        else:
            if num_clusters is None:
                return self._get_best_model(self._df, 2, 5)
            else:
                return self._train_model(self._df, num_clusters)

    def roc(self):
        # TODO: Pode usar curva ROC para validar modelo
        # - gráfico "true positives" por "false positives"
        # - Porcentagem TP = 100 * TP / (TP + FN)
        # - Porcentagem FP = 100 * FP / (FP + TN)
        return None

    def _get_best_model(self, df=None, min_clusters=2, max_clusters=5):
        if df is None:
            return None

        cv_results = []
        models = []
        for num_clusters in range(min_clusters, max_clusters + 1):
            models += [self._train_model(df, num_clusters)]

        # TODO: Best model logic (must use FPC and indexes of reference articles)
        return None

    def _cross_validation(self, model, df, num_folds):
        num_clusters = None  # TODO: Obtain number of clusters from somewhere
        frac = 1.0 / num_folds
        orig_df = df.copy()

        # Creating K folds
        folds = []
        for _ in range(num_folds):
            fold = orig_df.sample(frac=frac, random_state=self._seed)
            orig_df.drop(fold.index)
            folds += [fold]

        # Running cross validations
        results = []
        for test in folds:
            train = [fold for fold in folds if fold is not test]
            model = self._train_model(train, num_clusters)
            results += [self._test_model(model, test)]

        return results

    def _train_model(self, train, num_clusters):
        return fuzz.cmeans(data=train, c=num_clusters, m=2, error=0.005, maxiter=1000)

    def _test_model(self, model, test):
        return fuzz.cmeans_predict(test_data=test, cntr_trained=model[0], m=2, error=0.005, maxiter=1000)
