import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from sklearn.neighbors import KNeighborsClassifier

sys.path.append("../src")
from ellipsoid_classifier import EllipsoidClassifier
from leave_n_out_split import leave_n_out_split
from reduce_dimension import reduce_dimension_resize


def get_sample(df, label_col, label, n):
    return df[df[label_col] == label].sample(n)


def sample_to_numpy(sample, excluded_cols):
    return sample.loc[:, ~sample.columns.isin(excluded_cols)].to_numpy()


def rescale(X, scaling_factor):
    return np.array([scaling_factor * x for x in X])


def main():
    # Read the MNIST training data
    # The data in .csv form can be downloaded from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    data = pd.read_csv("mnist_train.csv", sep=',')

    # Define the main constants
    num_points = 600
    split_ratio = 1 / 3
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    new_dim = (10, 10)
    k = 15

    # Train test the classifier
    mod_svm_acc = []
    knn_acc = []
    excluded_labels_per_iter = []

    # Get num_points datapoints for each label
    for i in range(1, 6):
        X = []
        y = []

        for label in labels:
            X.append(sample_to_numpy(get_sample(data, 'label', label, num_points), ['label']).reshape((-1, 28, 28)))
            y.append([label] * num_points)

        X = np.concatenate(X)
        y = np.concatenate(y)

        X_reduced = reduce_dimension_resize(X, new_dim)
        X_reduced = rescale(X_reduced, 10**20).reshape(-1, new_dim[0] * new_dim[1])

        # Do the test train split leaving two labels out
        X_train, y_train, X_test, y_test = leave_n_out_split(X_reduced, y, 2, split_ratio=split_ratio)

        included_labels = np.unique(y_train)
        excluded_labels = list(set(labels).difference(set(included_labels)))
        excluded_labels_per_iter.append(excluded_labels)

        # Compute the means from a smaller sample for the left out labels
        n_few_shot = 50
        X_few_shot = []
        y_few_shot = []
        S = []
        y_S = []
        for label in excluded_labels:
            temp = sample_to_numpy(get_sample(data, 'label', label, n_few_shot), ['label']).reshape((-1, 28, 28))

            temp_reduced = reduce_dimension_resize(temp, new_dim)
            temp_reduced = rescale(temp_reduced, 10 ** 20).reshape(-1, new_dim[0] * new_dim[1])

            y_S.append(label)
            S.append(np.mean(temp_reduced, axis=0).astype(np.float))

            X_few_shot.append(temp_reduced)
            y_few_shot.append([label] * n_few_shot)

        X_few_shot = np.concatenate(X_few_shot)
        y_few_shot = np.concatenate(y_few_shot)

        model = EllipsoidClassifier(y_multiple=5)
        model.train(X_train, y_train, "banded", B=np.array([1, 10 - 1, 10, 10 + 1]))

        for label, node in model.nodes.items():
            y_S.append(label)
            S.append(node.center())

        S = np.array(S)
        y_S = np.array(y_S)

        knn = KNeighborsClassifier(k, weights="uniform").fit(np.concatenate([X_train, X_few_shot]),
                                                             np.concatenate([y_train, y_few_shot]))

        # Test the algorithms
        model.add_semantic_vectors(S, y_S)
        mod_svm_acc.append(model.predict(X_test, y_test, use_sem=True))

        y_hat = knn.predict(X_test)
        correct_classifications = [1 for y_actual, y_pred in zip(y_test, y_hat) if y_actual == y_pred]
        knn_acc.append(len(correct_classifications) / y_test.shape[0])

    print()
    print()
    print(mod_svm_acc)
    print(knn_acc)
    print(excluded_labels_per_iter)


if __name__ == '__main__':
    freeze_support()
    main()
