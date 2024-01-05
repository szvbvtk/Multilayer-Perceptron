import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split


def generate_dataset(m=1000, seed=None):
    rng = np.random.default_rng(seed)

    x1 = rng.uniform(0, np.pi, m)
    x2 = rng.uniform(0, np.pi, m)

    y = np.cos(x1 * x2) + np.cos(2 * x1)

    dataset = np.empty((m, 3))
    dataset[:, 0] = x1
    dataset[:, 1] = x2
    dataset[:, 2] = y

    return dataset


def draw_plots(mlp, X, y):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1_v, x2_v = np.meshgrid(x1, x2)
    Xv = np.column_stack((x1_v.ravel(), x2_v.ravel()))

    y_pred = mlp.predict(Xv)
    y_pred = y_pred.reshape(x1_v.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"}
    )

    ax1.scatter(X[:, 0], X[:, 1], y, c=y, cmap="cool_r", alpha=0.5)  # probki
    ax1.set_facecolor("lavender")
    ax1.set_title("Wykres zbioru próbek")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")

    ax2.draw_plots(x1_v, x2_v, y_pred, alpha=0.8, cmap="Spectral")
    ax2.set_facecolor("lavender")
    ax2.set_title("Wykres funkcji aproksymowanej")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("y")

    ax3.scatter(X[:, 0], X[:, 1], y, c=y, cmap="cool_r", alpha=0.1)
    ax3.draw_plots(x1_v, x2_v, y_pred, alpha=0.8, cmap="summer")
    ax3.set_facecolor("lavender")
    ax3.set_title("Wykres zbioru próbek i funkcji aproksymowanej")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_zlabel("y")

    plt.show()


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
        # hidden_output_with_bias = np.concatenate(([1], hidden_output))
        # output = np.dot(self.weights_output, hidden_output_with_bias)
        # lub
        output = self.weights_output[:, 0] + np.dot(
            self.weights_output[:, 1:], hidden_output
        )
        return hidden_output, output

    # wersja do oddania (rozumiem co sie tu dzieje)
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
    #                 * self.weights_output[0, k_+1]
    #                 * hidden_output[k_+1]
    #                 * (1 - hidden_output[k_+1])
    #                 * X[j_]
    #             )

    #     self.weights_output -= self.learning_rate * error * hidden_output

    # wersja do commita (dziala szybciej)
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

    def fit(self, X, y):
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

            print(f"Step: {step + 1} / {self.number_of_steps}")
            self.backward(X_sample, y_sample, hidden_output, output)

        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            _, y_pred[i] = self.forward(X[i, :])
        return y_pred


# dataset = generate_dataset(m=10000, seed=0)
# X, y = dataset[:, :-1], dataset[:, -1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# mlp = MultilayerPerceptronRegressor(
#     number_of_neurons=16, number_of_steps=10e4, learning_rate=0.05, seed=0
# )

# mlp.fit(X_train, y_train)

# y_pred = mlp.predict(X_test)
# mse = np.mean((y_test - y_pred) ** 2)
# print(f"Mean Squared Error: {mse}")
# print(mlp.weights_hidden)
# draw_plots(mlp, X_train, y_train)

# X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
# y = np.array([1, -1, -1, 1])

# mlp = MultilayerPerceptronRegressor(
#     number_of_neurons=16, number_of_steps=10e4, learning_rate=0.05, seed=0
# )
# mlp.fit(X, y)
# y_pred = mlp.predict(X)

# print(y_pred)
