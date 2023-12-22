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

def plot_surface(mlp, X, y):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1_v, x2_v = np.meshgrid(x1, x2)
    Xv = np.column_stack((x1_v.ravel(), x2_v.ravel()))
    # print(x1_v.shape, x2_v.shape, Xv.shape)
    y_pred = mlp.predict(Xv)
    y_pred = y_pred.reshape(x1_v.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

    scatter = ax1.scatter(X[:, 0], X[:, 1], y, c=y, cmap="cool_r", alpha=0.5)
    ax1.set_facecolor("lavender")
    ax1.set_title("Wykres zbioru próbek")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")

    surface = ax2.plot_surface(x1_v, x2_v, y_pred, alpha=0.8, cmap="Spectral")
    ax2.set_facecolor("lavender")
    ax2.set_title("Wykres funkcji aproksymowanej")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("y")

    ax3.scatter(X[:, 0], X[:, 1], y, c=y, cmap="cool_r", alpha=0.1)
    ax3.plot_surface(x1_v, x2_v, y_pred, alpha=0.8, cmap="summer")
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
        hidden_input = np.dot(self.weights_hidden, np.concatenate(([1], X)))
        hidden_output = self.sigmoid(hidden_input)
        output = np.dot(self.weights_output, np.concatenate(([1], hidden_output)))
        return hidden_input, hidden_output, output

    def backward(self, X, y, hidden_input, hidden_output, output):
        self.weights_output -= self.learning_rate * (
            (output - y) * np.concatenate(([1], hidden_output))
        )
        self.weights_hidden -= self.learning_rate * np.outer(
            np.dot((output - y), self.weights_output[:, 1:])
            * hidden_output
            * (1 - hidden_output),
            np.concatenate(([1], X)),
        )

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

            hidden_input, hidden_output, output = self.forward(X_sample)

            error = 0.5 * np.power((output - y_sample), 2)

            # if error < 1e-9:
            #     print(f"Achieved satisfactory error: {error}")
            #     print(f"Step: {step + 1} / {self.number_of_steps}, Error: {error}")
            #     break

            print(f"Step: {step + 1} / {self.number_of_steps}")
            self.backward(X_sample, y_sample, hidden_input, hidden_output, output)

        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            _, _, y_pred[i] = self.forward(X[i, :])
        return y_pred


dataset = generate_dataset(m=10000, seed=0)
X, y = dataset[:, :-1], dataset[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mlp = MultilayerPerceptronRegressor(
    number_of_neurons=16, number_of_steps=10e5, learning_rate=0.05, seed=0
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error: {mse}")

plot_surface(mlp, X_train, y_train)
