import numpy as np


class LinearRegressionGD(object):
    def __init__(self, fit_intercept=True, copy_X=True,
                 eta0=0.001, epochs=1000, learning_rate_decay=1,
                 shuffle=True, batch_size=128):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self._eta0 = eta0
        self._epochs = epochs

        self._cost_history = []

        self._coef = None
        self._intercept = None
        self._new_X = None
        self._w_history = None
        self._learning_rate_decay = learning_rate_decay

        self._shuffle = shuffle
        self._batch_size = batch_size

    def cost(self, h, y):
        return 1/(2*len(y)) * np.sum((h-y).flatten() ** 2)

    def hypothesis_function(self, X, theta):
        return X.dot(theta).reshape(-1, 1)

    def gradient(self, X, y, theta):
        return X.T.dot(self.hypothesis_function(X, theta)-y) / len(X)

    def fit(self, X, y):

        self._new_X = np.array(X)
        y = y.reshape(-1, 1)

        if self.fit_intercept:
            intercept_vector = np.ones([len(self._new_X), 1])
            self._new_X = np.concatenate(
                    (intercept_vector, self._new_X), axis=1)

        theta_init = np.random.normal(0, 5, self._new_X.shape[1])
        self._w_history = [theta_init]
        self._cost_history = [self.cost(
                        self.hypothesis_function(self._new_X, theta_init), y)]

        theta = theta_init

        for epoch in range(self._epochs):
            if self._shuffle:
                np.random.shuffle(self._new_X)
            n_batch = len(self._new_X) // self._batch_size
            for batch_count in range(n_batch):
                X_batch = self._new_X[batch_count*self._batch_size: \
                                (batch_count+1)*self._batch_size]
                y_batch = y[batch_count*self._batch_size: \
                                (batch_count+1)*self._batch_size]

                gradient = self.gradient(X_batch, y_batch, theta).flatten()
                theta = theta - self._eta0 * gradient

            if epoch % 10 == 0:
                self._w_history.append(theta)
                cost = self.cost(
                    self.hypothesis_function(self._new_X, theta), y)
                self._cost_history.append(cost)
            self._eta0 = self._eta0 * self._learning_rate_decay

        if self.fit_intercept:
            self._intercept = theta[0]
            self._coef = theta[1:]
        else:
            self._coef = theta

    def predict(self, X):
        """
        적합된 Linear regression 모델을 사용하여 입력된 Matrix X의 예측값을 반환한다.
        이 때, 입력된 Matrix X는 별도의 전처리가 없는 상태로 입력되는 걸로 가정한다.
        fit_intercept가 True일 경우:
            - Matrix X의 0번째 Column에 값이 1인 column vector를추가한다.
        Parameters
        ----------
        X : numpy array, 2차원 matrix 형태로 [n_samples,n_features] 구조를 가진다

        Returns
        -------
        y : numpy array, 예측된 값을 1차원 vector 형태로 [n_predicted_targets]의
            구조를 가진다.
        """
        test_X = np.array(X)

        if self.fit_intercept:
            intercept_vector = np.ones([len(test_X), 1])
            test_X = np.concatenate(
                    (intercept_vector, test_X), axis=1)

            weights = np.concatenate(([self._intercept], self._coef), axis=0)
        else:
            weights = self._coef

        return test_X.dot(weights)

    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept

    @property
    def weights_history(self):
        return np.array(self._w_history)

    @property
    def cost_history(self):
        return self._cost_history
