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
    scaler = scaling_factor * np.identity(X.shape[1])
    return np.array([np.matmul(scaler, x) for x in X])


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

    # Visualize the dimensional reduction
    examples = []
    for label in labels:
        label_df = data[data["label"] == label]
        label_data = label_df.drop(["label"], axis=1).to_numpy().reshape(-1, 28, 28)
        examples.append(label_data[0])

    examples = np.array(examples)

    # Invert the colors
    examples = 100 - examples

    # Reduce the dimension for the examples
    reduced_examples = reduce_dimension_resize(examples, new_dim)

    fig, ax = plt.subplots(nrows=2, ncols=len(labels))

    ex_min = np.min([np.min(e) for e in examples])
    ex_max = np.max([np.max(e) for e in examples])

    red_min = np.min([np.min(e) for e in reduced_examples])
    red_max = np.max([np.max(e) for e in reduced_examples])

    for i in range(0, len(examples)):
        im0 = ax[0, i].imshow(examples[i], vmin=ex_min, vmax=ex_max, cmap="Greys")
        im1 = ax[1, i].imshow(reduced_examples[i], vmin=red_min, vmax=red_max, cmap="Greys")
        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.show()

    # Train test the classifier
    mod_svm_acc = []
    knn_acc = []

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

        # Do the test train split
        X_train, y_train, X_test, y_test = leave_n_out_split(X_reduced, y, 0, split_ratio=split_ratio)

        model = EllipsoidClassifier(y_multiple=5)
        model.train(X_train, y_train, "banded", B=np.array([1, 10 - 1, 10, 10 + 1]))

        knn = KNeighborsClassifier(k, weights="uniform").fit(X_train, y_train)

        # Test the algorithms with semantic vectors and without
        mod_svm_acc.append(model.predict(X_test, y_test))

        y_hat = knn.predict(X_test)
        correct_classifications = [1 for y_actual, y_pred in zip(y_test, y_hat) if y_actual == y_pred]
        knn_acc.append(len(correct_classifications) / y_test.shape[0])

    print()
    print()
    print(mod_svm_acc)
    print(knn_acc)


if __name__ == '__main__':
    freeze_support()
    main()
