import numpy as np
from random import shuffle, sample


def leave_n_out_split(X, y, n,
                      split_ratio=0.25,
                      test_only_with_excluded=False):
    """
    Relatively generic test-train split function with the added functionality for leaving out some labels from
    the training set
    :param X: numpy array of shape (n, d) containing all of the data available
    :param y: numpy array of shape (n, ) containing the labels corresponding with data
    :param n: int telling the number of labels to be left out of the training
    :param split_ratio: Optional. float telling the percentage of datapoints left for testing. Defaults to 0.25
    :param test_only_with_excluded: Optional. bool telling whether the testing set should consist of points of only
    the label that were left out of the training. Defaults to False
    :return: numpy arrays corresponding with training data and labels and testing data and labels
    in order X_train, y_train, X_test, y_test
    """
    assert X.shape[0] == y.shape[0]

    # Get the distinct labels
    labels = np.unique(y).tolist()
    assert n < len(labels)

    # Using random sampling split the data into two s.t. n distinct labels have been left out of training data
    excluded_labels = sample(labels, n)

    # Test train split if we are only interested in zero shot capabilities
    if test_only_with_excluded:
        # Get the indexes of the sampled labels in y
        excluded_indexes = [i for i in range(len(y)) if y[i] in excluded_labels]

        # Get all other indexes
        included_indexes = [i for i in range(len(y)) if y[i] not in excluded_labels]

        X_test = X[excluded_indexes]
        X_train = X[included_indexes]
        y_test = y[excluded_indexes]
        y_train = y[included_indexes]

        return X_train, y_train, X_test, y_test

    # More traditional test train split
    else:
        training_indexes = []
        testing_indexes = []

        for label in labels:
            if label not in excluded_labels:
                print(f"Included label: {label}")
                label_indexes = [i for i in range(len(y)) if y[i] == label]
                shuffle(label_indexes)
                split_index = int(split_ratio * len(label_indexes))
                testing_indexes.append(label_indexes[0:split_index])
                training_indexes.append(label_indexes[split_index:len(label_indexes)])
            else:
                print(f"Excluded label: {label}")
                label_indexes = [i for i in range(len(y)) if y[i] == label]
                shuffle(label_indexes)
                split_index = int(split_ratio * len(label_indexes))
                testing_indexes.append(label_indexes[0:split_index])

        training_indexes = np.concatenate(training_indexes, axis=0).reshape(-1,)
        testing_indexes = np.concatenate(testing_indexes, axis=0).reshape(-1,)

        X_test = X[testing_indexes]
        X_train = X[training_indexes]
        y_test = y[testing_indexes]
        y_train = y[training_indexes]

        return X_train, y_train, X_test, y_test
