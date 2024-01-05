from MultilayerPerceptron import (
    generate_dataset,
    draw_plots,
    MultilayerPerceptronRegressor,
)
import numpy as np


def test():
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    y = np.array([1, -1, -1, 1])

    mlp = MultilayerPerceptronRegressor(
        number_of_neurons=16, number_of_steps=10e4, learning_rate=0.05, seed=0
    )

    mlp.fit(X, y)
    y_pred = mlp.predict(X)
    print(f"y = {y}\ny_pred = {y_pred}")


def main1():
    dataset = generate_dataset(m=10000, seed=0)
    X, y = dataset[:, :-1], dataset[:, -1]

    mlp = MultilayerPerceptronRegressor(
        number_of_neurons=16, number_of_steps=10e4, learning_rate=0.05, seed=0
    )

    mlp.fit(X, y)
    draw_plots(mlp, X, y)


def main2():
    pass


if __name__ == "__main__":
    test()
