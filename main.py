from MultilayerPerceptron import MultilayerPerceptronRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
from tabulate import tabulate as tab
import seaborn as sns


def generate_dataset(m=1000, seed=None, error=False):
    rng = np.random.default_rng(seed)

    x1 = rng.uniform(0, np.pi, m)
    x2 = rng.uniform(0, np.pi, m)

    y = np.cos(x1 * x2) * np.cos(2 * x1)

    if error:
        y += rng.normal(0, 0.2, m)

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

    _, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=(15, 5), subplot_kw={"projection": "3d"}
    )

    ax1.scatter(X[:, 0], X[:, 1], y, c=y, cmap="cool_r", alpha=0.5)  # probki
    ax1.set_facecolor("lavender")
    ax1.set_title("Wykres zbioru próbek")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")

    y = np.cos(x1_v * x2_v) * np.cos(2 * x1_v)
    ax2.plot_surface(x1_v, x2_v, y, alpha=0.8, cmap="OrRd")
    ax2.set_facecolor("lavender")
    ax2.set_title("Wykres funkcji")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("y")

    ax3.plot_surface(x1_v, x2_v, y_pred, alpha=0.8, cmap="YlGn")
    ax3.set_facecolor("lavender")
    ax3.set_title("Wykres sieci")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_zlabel("y")

    # ax4.scatter(X[:, 0], X[:, 1], y, c=y, cmap="cool_r", alpha=0.1)
    ax4.plot_surface(x1_v, x2_v, y, alpha=0.8, cmap="OrRd")
    ax4.plot_surface(x1_v, x2_v, y_pred, alpha=0.8, cmap="YlGn")
    ax4.set_facecolor("lavender")
    ax4.set_title("Wykres funkcji i sieci")
    ax4.set_xlabel("x1")
    ax4.set_ylabel("x2")
    ax4.set_zlabel("y")

    plt.show()


def test(verbose=True):
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    y = np.array([1, -1, -1, 1])

    mlp = MultilayerPerceptronRegressor(
        number_of_neurons=16, number_of_steps=1e5, learning_rate=0.05, seed=0
    )

    mlp.fit(X, y, verbose=verbose)
    y_pred = mlp.predict(X)
    print(f"y = {y}\ny_pred = {y_pred}")


def main1(verbose=True, number_of_steps=1e5):
    dataset = generate_dataset(m=10000, seed=0)
    X, y = dataset[:, :-1], dataset[:, -1]

    mlp = MultilayerPerceptronRegressor(
        number_of_neurons=16,
        number_of_steps=number_of_steps,
        learning_rate=0.05,
        seed=0,
    )

    mlp.fit(X, y, verbose=verbose)
    draw_plots(mlp, X, y)


def main2(verbose=True, number_of_steps=1e5):
    dataset = generate_dataset(m=200, seed=0, error=True)
    X, y = dataset[:, :-1], dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    number_of_neurons_list = [i for i in range(10, 101, 10)]
    train_errors = np.empty_like(number_of_neurons_list, dtype=np.float16)
    test_errors = np.empty_like(number_of_neurons_list, dtype=np.float16)

    for i, number_of_neurons in enumerate(number_of_neurons_list):
        mlp = MultilayerPerceptronRegressor(
            number_of_neurons=number_of_neurons,
            number_of_steps=number_of_steps,
            learning_rate=0.05,
            seed=0,
        )

        mlp.fit(X_train, y_train, verbose=verbose)

        y_train_pred = mlp.predict(X_train)
        y_test_pred = mlp.predict(X_test)

        train_errors[i] = mean_absolute_error(y_train_pred, y_train)
        test_errors[i] = mean_absolute_error(y_test_pred, y_test)

    df = pd.DataFrame(
        {
            "Number of neurons": number_of_neurons_list,
            "Train error": train_errors,
            "Test error": test_errors,
        }
    )

    print(tab(df, headers="keys", showindex=False, tablefmt="fancy_grid"))

    sns.set_theme(style="whitegrid", palette="deep", rc={"axes.facecolor": "beige"})

    df_for_plot = pd.melt(
        frame=df,
        id_vars="Number of neurons",
        value_vars=["Train error", "Test error"],
        var_name="Dataset",
        value_name="Mean absolute error",
    )

    sns.barplot(
        x="Number of neurons",
        y="Mean absolute error",
        hue="Dataset",
        data=df_for_plot,
        palette=("orangered", "darkcyan"),
    )

    plt.axhline(
        y=np.min(df_for_plot["Mean absolute error"]),
        color="rebeccapurple",
        label="Optimal number of neurons",
        linestyle=(5, (10, 3)),
    )

    plt.xlabel("Number of neurons")
    plt.ylabel("Mean absolute error")
    plt.yticks(np.arange(0, np.max(df_for_plot["Mean absolute error"]) + 0.2, 0.2))

    optimal_number_of_neurons = df["Number of neurons"][np.argmin(test_errors)]

    handles, labels = plt.gca().get_legend_handles_labels()
    labels[0] += f": {optimal_number_of_neurons}"
    plt.legend(handles, labels)

    plt.show()

    print(f"Optimal number of neurons: {optimal_number_of_neurons}")
    mlp = MultilayerPerceptronRegressor(
        number_of_neurons=optimal_number_of_neurons,
        number_of_steps=number_of_steps,
        learning_rate=0.05,
        seed=0,
    )


if __name__ == "__main__":
    # test()
    # main1()
    main2(False, 1e5)
