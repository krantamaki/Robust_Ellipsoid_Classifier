import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from sklearn.neighbors import KNeighborsClassifier

sys.path.append("../src")
from ellipsoid_classifier import EllipsoidClassifier
from leave_n_out_split import leave_n_out_split


COLORS = ["gray", "lightcoral", "firebrick", "red", "coral", "sienna", "peru", "darkorange", "goldenrod", "darkkhaki",
          "olive", "yellow", "green", "lime", "turquoise", "teal", "steelblue", "navy", "blue", "blueviolet", "purple",
          "brown", "sandybrown", "orange", "darkseagreen", "seagreen", "darkslategrey", "midnightblue", "indigo",
          "violet", "crimson", "pink"]

# Means and standard deviations of given labels
MEANS = [(1., 1.), (1., 2.), (1.5, 1.5), (2., 2.), (0.5, 1.), (0.5, 0.5), (3., 1.), (1.5, 0.5), (3., 2.), (2., 1.),
         (0.5, 2.), (1., 0.5), (2., 0.5), (2.5, 2.5), (1., 1.5), (1.5, 1.), (1.5, 2.5), (2.5, 1.5), (1., 0.), (2.5, 0.),
         (0., 1.), (2., 0.), (3.5, 0), (0.5, 2.5), (3., 1.5), (3., 0.5), (2, 2.5), (0., 2.5), (3., 0.), (3.5, 1.5)]

DEVIATIONS = [(0.2, 0.1), (0.3, 0.2), (0.3, 0.2), (0.3, 0.1), (0.2, 0.3), (0.2, 0.2), (0.2, 0.25), (0.2, 0.3),
              (.4, 0.2), (0.15, 0.2),
              (0.2, 0.1), (0.15, 0.15), (0.1, 0.3), (0.1, 0.2), (0.3, 0.2), (0.1, 0.1), (0.15, 0.2), (0.2, 0.2),
              (0.15, 0.25), (0.3, 0.1),
              (0.2, 0.2), (0.1, 0.1), (0.3, 0.3), (0.2, 0.1), (0.3, 0.05), (0.2, 0.1), (0.2, 0.1), (0.3, 0.2),
              (0.1, 0.1), (0.1, 0.1)]


def gen_points(mean, deviation, n):
    x = np.random.normal(mean[0], deviation[0], size=n)
    y = np.random.normal(mean[1], deviation[1], size=n)
    return np.column_stack((x, y))


def gen_sem_vector(mean):
    return np.array([mean[0] ** 2 - mean[0], mean[0] ** (1 / 2) + mean[1] ** (3 / 2),
                     2 * mean[1] - mean[0] ** 2, mean[1] ** 3])
    # return np.array([mean[0] ** 2 - mean[0], mean[0] ** (1 / 2),
    #                  2 * mean[1] - mean[1] ** (1 / 2), mean[1] ** 3])
    # return np.array([mean[0] ** 2, mean[0] ** (1 / 2),
    #                  mean[1] ** (1 / 2), mean[1] ** 2])
    # return np.array([3 * mean[0] + 3.2, 0.761 * mean[0],
    #                  0.2335 * mean[1], 7 * mean[1] + 1.245])
    # return np.array([mean[0] ** 2, mean[1] ** 2])


def main():
    # How many points are wanted per label
    num_points = 50
    # How many labels are used
    num_labels = 30
    # How many neighbors are evaluated by the k-nearest neighbor algorithm
    k = 15

    labels = [f"$\mu$: {MEANS[i]}, $\sigma^2$: {DEVIATIONS[i]}" for i in range(num_labels)]

    X = np.concatenate([gen_points(MEANS[i], DEVIATIONS[i], num_points) for i in range(num_labels)])
    y = np.concatenate([[labels[i]] * num_points for i in range(num_labels)])

    # Visualize the datapoints
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot()
    for i in range(num_labels):
        ax.scatter(X[:, 0][i * num_points:i * num_points + num_points // 2],
                   X[:, 1][i * num_points:i * num_points + num_points // 2],
                   color=COLORS[i],
                   s=20,
                   label=labels[i])

    # ax.legend()
    plt.tight_layout()
    plt.show()

    S = np.array([gen_sem_vector(MEANS[i]) for i in range(num_labels)])
    sem_y = np.array([f"$\mu$: {MEANS[i]}, $\sigma^2$: {DEVIATIONS[i]}" for i in range(num_labels)])

    kneighbors_mean = []
    kneighbors_std = []
    sem_mean = []
    sem_std = []
    base_mean = []
    base_std = []

    # Go from leave 0 out until leave 10 out
    for i in range(0, 10 + 1):
        kneighbors_tmp = []
        sem_tmp = []
        base_tmp = []

        # As the left out labels are chosen randomly do the test 10 times and average the results together
        for _ in range(0, 10):
            # Do test train split
            X_train, y_train, X_test, y_test = leave_n_out_split(X, y, i, test_only_with_excluded=False, split_ratio=0.5)
            print()

            # Train the models
            model = EllipsoidClassifier(y_multiple=5)
            model.train(X_train, y_train, "ind")

            knn = KNeighborsClassifier(k, weights="uniform").fit(X_train, y_train)

            # Pass semantic information to the model
            model.add_semantic_vectors(S, sem_y)

            # Test the algorithms with semantic vectors and without
            sem_tmp.append(model.predict(X_test, y_test, use_sem=True))
            base_tmp.append(model.predict(X_test, y_test, use_sem=False))

            y_hat = knn.predict(X_test)
            correct_classifications = [1 for y_actual, y_pred in zip(y_test, y_hat) if y_actual == y_pred]
            kneighbors_tmp.append(len(correct_classifications) / y_test.shape[0])

        kneighbors_mean.append(np.array([kneighbors_tmp]).mean())
        kneighbors_std.append(np.array([kneighbors_tmp]).std())
        sem_mean.append(np.array([sem_tmp]).mean())
        sem_std.append(np.array([sem_tmp]).std())
        base_mean.append(np.array([base_tmp]).mean())
        base_std.append(np.array([base_tmp]).std())

    # Plot as barplot
    fig = plt.figure(figsize=[10, 6])
    x_axis = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    plt.errorbar(x_axis, np.array(sem_mean) * 100, np.array(sem_std) * 100, marker='o', capsize=3,
                 label='Our algorithm with semantic vectors')
    plt.errorbar(x_axis, np.array(base_mean) * 100, np.array(base_std) * 100, marker='o', capsize=3,
                 label='Our algorithm without semantic vectors')
    plt.errorbar(x_axis, np.array(kneighbors_mean) * 100, np.array(kneighbors_std) * 100, marker='o', capsize=3,
                 label='k-Nearest Neighbor')

    # plt.xticks(x_axis, x_axis)
    plt.xlabel("Number of labels left out of the training data")
    plt.ylabel("Prediction accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    freeze_support()
    main()
