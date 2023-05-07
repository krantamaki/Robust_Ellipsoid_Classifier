from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

colors = ['red', 'blue', 'green', 'grey', 'magenta', 'cyan', 'orange', 'deeppink', 'brown', 'darkviolet']


def viz_knn(y, X, labels, k_neighbors,
            dev_extrema=1,
            axis_names=("x", "y"),
            verbose=True,
            show=True,
            save_path="figures/kneighbors.png"):
    # Train the model
    knn = KNeighborsClassifier(k_neighbors, weights="uniform").fit(X, y)

    # Get the maxima and minima for the columns of X
    maxima = np.max(X, axis=0)
    minima = np.min(X, axis=0)

    # Create bounds for data to find the decision boundary
    num_points = 300
    xx = np.linspace(minima[0] - dev_extrema, maxima[0] + dev_extrema, num_points)
    yy = np.linspace(minima[1] - dev_extrema, maxima[1] + dev_extrema, num_points)

    # Use the trained model to predict labels for points defined above
    point_xs = []
    point_ys = []
    predictions = []
    count = 0
    for point_x in xx:
        for point_y in yy:
            point_xs.append(point_x)
            point_ys.append(point_y)
            predictions.append(knn.predict(np.array([point_x, point_y]).reshape(1, -1)))

        if verbose:
            print(f"{(count / num_points) * 100}% of mapping done")
        count += 1

    # Generate color for each label
    label_to_color = {}
    if len(labels) <= 10:
        for i, label in enumerate(labels):
            label_to_color[label] = colors[i]

    else:
        raise ValueError("Number of distinct labels exceeds 10")

    prediction_colors = [label_to_color[label[0]] for label in predictions]

    # Plot everything
    if verbose:
        print("\nDrawing figure...")

    fig, ax = plt.subplots()

    ax.scatter(point_xs, point_ys, c=prediction_colors, s=50, alpha=0.01, lw=0)

    # Sort the data points by label
    tuples = [(y[i], X[i][0], X[i][1]) for i in range(y.shape[0])]
    for label in labels:
        data_points = np.array([[tup[1], tup[2]] for tup in tuples if tup[0] == label])
        ax.scatter(data_points[:, 0], data_points[:, 1], c=label_to_color[label], s=50,
                   edgecolor="white", linewidth=1, label=label)

    ax.set(xlim=(minima[0] - 1, maxima[0] + 1), ylim=(minima[1] - 1, maxima[1] + 1),
           xlabel=axis_names[0], ylabel=axis_names[1])

    ax.legend(loc='lower right')

    if show:
        plt.show()
    else:
        fig.savefig(save_path)


def gen_points(mean, deviation, n):
    x = np.random.normal(mean[0], deviation[0], size=n)
    y = np.random.normal(mean[1], deviation[1], size=n)
    return np.column_stack((x, y))


def main():
    # Means and standard deviations for given labels
    means = [(1., 1.), (1., 2.), (2., 1.), (2., 2.)]
    deviations = [(0.12, 0.08), (0.13, 0.17), (0.16, 0.11), (0.16, 0.07)]

    n = 50
    k = 15

    labels = [f"$\mu$: {means[i]}, $\sigma^2$: {deviations[i]}" for i in range(len(means))]

    X = np.concatenate([gen_points(means[i], deviations[i], n) for i in range(len(means))])
    y = np.concatenate([[labels[i]] * n for i in range(len(means))])

    viz_knn(y, X, labels, k)


if __name__ == "__main__":
    main()
