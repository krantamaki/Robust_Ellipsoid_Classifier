import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from multiprocessing import freeze_support

sys.path.append("../src")
from ellipsoid_classifier import EllipsoidClassifier

colors = ['red', 'blue', 'green', 'grey', 'magenta', 'cyan', 'orange', 'deeppink', 'brown', 'darkviolet']


def viz_ellipsoid(y0, X0, labels, classifier,
                  axis_names=("x", "y"),
                  show=True,
                  save_path="figures/ellipsoids.png"):

    # Get the maxima and minima for the columns of X
    maxima = np.max(X0, axis=0)
    minima = np.min(X0, axis=0)

    max_limit = max(maxima)
    min_limit = min(minima)

    # Generate color for each label
    label_to_color = {}
    if len(labels) <= 10:
        for i, label in enumerate(labels):
            label_to_color[label] = colors[i]
    else:
        raise ValueError("Number of distinct labels exceeds 10")

    fig, ax = plt.subplots()

    # Plot the circles representing the region allocated for each node
    for node in classifier.nodes.values():
        label_color = label_to_color[node.label]
        width = 2 * node.matrix()[0, 0] ** (-1 / 2)
        height = 2 * node.matrix()[1, 1] ** (-1 / 2)
        # width = node.matrix()[0, 0]
        # height = node.matrix()[1, 1]
        ellipse = Ellipse(node.center(), width=width, height=height,
                          color=label_color, alpha=0.3, linewidth=1)
        ax.add_patch(ellipse)

    # Sort the data points by label
    tuples = [(y0[i], X0[i][0], X0[i][1]) for i in range(y0.shape[0])]
    for label in labels:
        data_points = np.array([[tup[1], tup[2]] for tup in tuples if tup[0] == label])
        ax.scatter(data_points[:, 0], data_points[:, 1], c=label_to_color[label], s=50,
                   edgecolor="white", linewidth=1, label=label)

    ax.set(xlim=(min_limit - 1, max_limit + 1), ylim=(min_limit - 1, max_limit + 1),
           xlabel=axis_names[0], ylabel=axis_names[1])

    ax.legend(loc='lower right')

    if show:
        plt.show()
    else:
        fig.savefig(save_path)


def viz_boundaries(y, X, labels, classifier,
                   use_sem=False,
                   axis_names=("x", "y"),
                   dev_extrema=1,
                   verbose=True,
                   show=True,
                   save_path="figures/boundaries.png"):

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
            predictions.append(classifier.predict(np.array([point_x, point_y]).reshape(1, -1), use_sem=use_sem))

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
    means = [(0.5, 2.), (1, 1.75), (1.5, 1.25), (2., 1.)]
    deviations = [(0.12, 0.08), (0.13, 0.17), (0.16, 0.11), (0.16, 0.07)]
    # means = [(1., 1.), (1., 2.), (2., 1.), (2., 2.), (1.5, 2.5), (0.4, 1.6), (0.5, 0.8), (2., 1.6)]
    # deviations = [(0.12, 0.08), (0.13, 0.17), (0.16, 0.11), (0.16, 0.07), (0.11, 0.1), (0.03, 0.05), (0.12, 0.19), (0.07, 0.1)]

    n = 50

    labels = [f"$\mu$: {means[i]}, $\sigma^2$: {deviations[i]}" for i in range(len(means))]

    X0 = np.concatenate([gen_points(means[i], deviations[i], n) for i in range(len(means))])
    y0 = np.concatenate([[labels[i]] * n for i in range(len(means))])

    model = EllipsoidClassifier(y_multiple=4)
    model.train(X0, y0, "ind")

    viz_ellipsoid(y0, X0, labels, model)

    # Unseen label
    # mean_unseen = (1.5, 1.5)
    # dev_unseen = (0.1, 0.1)
    unseen_means = [(0.5, 0.5)]  # , (0.75, 1.5)]
    unseen_deviations = [(0.1, 0.11)]  # , (0.12, 0.06)]
    unseen_labels = [f"$\mu$: {unseen_means[i]}, $\sigma^2$: {unseen_deviations[i]}" for i in range(len(unseen_means))]

    X = np.concatenate([X0, *[gen_points(unseen_means[i],unseen_deviations[i], n) for i in range(len(unseen_means))]])
    y = np.concatenate([y0, *[[unseen_labels[i]] * n for i in range(len(unseen_means))]])
    labels += unseen_labels

    S = np.array(means + unseen_means)
    s_y = np.array(labels)

    model.add_semantic_vectors(S, s_y)

    viz_boundaries(y, X, labels, model, use_sem=True)


if __name__ == "__main__":
    freeze_support()
    main()
