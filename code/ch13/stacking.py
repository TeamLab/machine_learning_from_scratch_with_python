from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.base import BaseEstimator

from sklearn.model_selection import train_test_split

import numpy as np


class MyStackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, meta_estimator, base_estimators, test_ratio=0.2, feature_weights=None):
        """
        Called when initializing the classifier
        """
        self.meta_estimator = meta_estimator
        self.base_estimators = base_estimators
        self.feature_weights = feature_weights
        self.test_ratio = test_ratio

    def fit(self, X, y=None):

        X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=self.test_ratio)

        for estimator in self.base_estimators:
            estimator.fit(X_train, y_train)

        meta_train_set = np.array([estimator.predict(X_test)
                                   for estimator in self.base_estimators]).T


        self.meta_estimator.fit(meta_train_set, y_test)

        return self

    def predict(self, X, y=None):
        meta_X = []
        for estimator in self.base_estimators:
            meta_X.append(estimator.predict(X))
        meta_X = np.array(meta_X).T

        return self.meta_estimator.predict(meta_X)


class MyStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, meta_estimator, base_estimators, test_ratio=0.2, feature_weights=None):
        """
        Called when initializing the classifier
        """
        self.meta_estimator = meta_estimator
        self.base_estimators = base_estimators
        self.feature_weights = feature_weights
        self.test_ratio = test_ratio

    def fit(self, X, y=None):

        X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=self.test_ratio)

        for estimator in self.base_estimators:
            estimator.fit(X_train, y_train)

        meta_train_set = np.array([estimator.predict(X_test)
                                   for estimator in self.base_estimators]).T


        self.meta_estimator.fit(meta_train_set, y_test)
        return self

    def predict(self, X, y=None):
        meta_X = []
        for estimator in self.base_estimators:
            meta_X.append(estimator.predict(X))
        meta_X = np.array(meta_X).T
        return self.meta_estimator.predict(meta_X)
