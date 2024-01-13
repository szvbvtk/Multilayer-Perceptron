import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MultilayerPerceptronRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, number_of_neurons, number_of_steps, learning_rate, seed):
        self.number_of_neurons = int(number_of_neurons)
        self.number_of_steps = int(number_of_steps)
        self.learning_rate = learning_rate
        self.seed = int(seed)
        self.weights_hidden = None
        self.weights_output = None

        # if self.seed is not None:
        self.rng = np.random.default_rng(self.seed)

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def forward(self, X):
        X = np.concatenate(([1], X))
        hidden_input = 1 + np.dot(self.weights_hidden, X)
        hidden_output = self.sigmoid(hidden_input)

        output = self.weights_output[:, 0] + np.dot(
            self.weights_output[:, 1:], hidden_output
        )
        return hidden_output, output

    # def backward(self, X, y, hidden_output, output):
    #     X = np.concatenate(([1], X))
    #     hidden_output = np.concatenate(([1], hidden_output))
    #     error = output - y
    #     k, j = self.weights_hidden.shape
    #     for k_ in range(k):
    #         for j_ in range(j):
    #             self.weights_hidden[k_, j_] -= (
    #                 self.learning_rate
    #                 * error
    #                 * self.weights_output[0, k_ + 1]
    #                 * hidden_output[k_ + 1]
    #                 * (1 - hidden_output[k_ + 1])
    #                 * X[j_]
    #             )

    #     self.weights_output -= self.learning_rate * error * hidden_output

    def backward(self, X, y, hidden_output, output):
        X = np.concatenate(([1], X))
        error = output - y

        self.weights_hidden -= self.learning_rate * np.outer(
            np.dot(error, self.weights_output[:, 1:])
            * hidden_output
            * (1 - hidden_output),
            X,
        )

        hidden_output = np.concatenate(([1], hidden_output))
        self.weights_output -= self.learning_rate * error * hidden_output

    def fit(self, X, y, verbose=True):
        rng_min = -1e-3
        rng_max = 1e-3
        number_of_features = X.shape[1]
        self.weights_hidden = self.rng.uniform(
            rng_min, rng_max, (self.number_of_neurons, number_of_features + 1)
        )
        self.weights_output = self.rng.uniform(
            rng_min, rng_max, (1, self.number_of_neurons + 1)
        )

        for step in range(self.number_of_steps):
            random_index = self.rng.integers(0, X.shape[0])
            X_sample = X[random_index, :]
            y_sample = y[random_index]

            hidden_output, output = self.forward(X_sample)

            if verbose:
                print(f"Step: {step + 1} / {self.number_of_steps}")

            self.backward(X_sample, y_sample, hidden_output, output)

        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            _, y_pred[i] = self.forward(X[i, :])
        return y_pred
