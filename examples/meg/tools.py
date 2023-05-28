"""
Potentially useful helper functions
"""
import os
import random
import torch
import numpy as np
from scipy.io import loadmat

from config import *


def rescale(X, scaling_factor):
    scaler = scaling_factor * np.ones((X.shape[1],))
    return np.array([scaler * x for x in X])


def form_dataset():
    """
    Function for forming the training and testing datasets and saving them in .npy format

    Returns: Void
    """
    print("\nForming the dataset...\n")

    # Randomly exclude one label from each category
    excluded_labels = []
    for cat in label_categories:
        excluded_labels.append(random.choice(label_categories[cat]))

    print(f"Excluded labels are: {', '.join(excluded_labels)}\n")

    # Gather the data into dictionaries of form label -> list[np.ndarray]
    train_dict = dict(zip(list(set(distinct_labels.copy()) - set(excluded_labels)),
                          [[] for _ in range(len(distinct_labels) - len(excluded_labels))]))
    test_dict = dict(zip(distinct_labels, [[] for _ in range(len(distinct_labels))]))

    for sub in range(n_subjects):
        subject_id = "sub-{:02d}".format(sub + 1)
        print(f"Processing subject: {subject_id}")
        if subject_id in excluded_subjects:
            continue

        data_path = f"{subject_path}/{subject_id}/epoch_data/{subject_id}_modality-{stimuli}_epoch_data.npy"
        data = np.load(data_path)

        # NOTE! The MEG data has shape (60, 18, 204, 40), where 60 is the number of words, 18 the number of reps, 204
        # the number of gradiometers and 40 the number of measurements per sensor. These will be changed to shape
        # (60, 18, 204 * 40)

        # Go over the nouns
        for i in range(0, n_words):
            label = distinct_labels[i]

            if label in excluded_labels:
                reps = np.array([data[i][rep].flatten() for rep in range(0, n_reps)])
                np.random.shuffle(reps)
                test_dict[label].append(rescale(reps[2 * (n_reps // 3):], 10 ** 12))
            else:
                reps = np.array([data[i][rep].flatten() for rep in range(0, n_reps)])
                np.random.shuffle(reps)
                train_dict[label].append(rescale(reps[:2 * (n_reps // 3)], 10 ** 12))
                test_dict[label].append(rescale(reps[2 * (n_reps // 3):], 10 ** 12))

    print()

    # We want to average together multiple repetitions for each datapoint
    # This means that the points aren't necessarily independent, but at least
    # they should be cleaner and this way we get more of them.
    # The averaging is done randomly so that 5 reps are averaged to form 1 datapoint
    # Number of final datapoints is set at 53000 for training set (1000 points per label)
    # and 12000 for testing set (200 points per label)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    train_ind = np.arange(0, 2 * (n_reps // 3) * (n_subjects - len(excluded_subjects)), 1)
    test_ind = np.arange(0, (n_reps // 3) * (n_subjects - len(excluded_subjects)), 1)

    for label in distinct_labels:
        print(f"Generating datapoints for label: {label}")
        if label in excluded_labels:
            test_data = np.concatenate(test_dict[label])
            for i in range(200):
                test_batch = np.random.choice(len(test_ind), size=n_averaged, replace=False)
                X_test.append(np.average(test_data[test_batch], axis=0))
                y_test.append(label)
        else:
            test_data = np.concatenate(test_dict[label])
            for i in range(200):
                test_batch = np.random.choice(len(test_ind), size=n_averaged, replace=False)
                X_test.append(np.average(test_data[test_batch], axis=0))
                y_test.append(label)

            train_data = np.concatenate(train_dict[label])
            for i in range(1000):
                train_batch = np.random.choice(len(train_ind), size=n_averaged, replace=False)
                X_train.append(np.average(train_data[train_batch], axis=0))
                y_train.append(label)

    print("\nSaving data...\n")

    np.save("train.npy", np.array(X_train))
    np.save("train_labels.npy", np.array(y_train))
    np.save("test.npy", np.array(X_test))
    np.save("test_labels.npy", np.array(y_test))


def load_dataset():
    """
    Loads the dataset formed by form_dataset function and returns it as torch.Tensor

    Returns: (X_train: torch.Tensor, y_train: list, X_test: torch.Tensor, y_test: list)
    """
    assert os.path.exists("train.npy")
    assert os.path.exists("train_labels.npy")
    assert os.path.exists("test.npy")
    assert os.path.exists("test_labels.npy")

    X_train = torch.Tensor(np.load("train.npy"))
    y_train = np.load("train_labels.npy").tolist()
    X_test = torch.Tensor(np.load("test.npy"))
    y_test = np.load("test_labels.npy").tolist()

    return X_train, y_train, X_test, y_test


def load_word2vec():
    """
    Function for loading the word2vec vectors

    Returns: word2vec_dict: dict[str, np.ndarray]
    """
    m_semantic = loadmat(word2vec_path)
    S = m_semantic['vectors']

    return dict(zip(distinct_labels, S))


def next_batch(data: torch.Tensor, labels: list, word2vecs: dict, batch_size: int, last: int):
    """
    Function for iterating over a tensor in mini-batches

    Returns: (mini_batch: torch.Tensor, batch_word2vec: torch.Tensor, batch_labels: list, new_last: int)
    """
    n_dims = data.shape[1]
    mini_batch = torch.zeros(batch_size, n_dims)
    batch_labels = []
    batch_word2vec = torch.zeros(batch_size, word2vec_dim)

    for i0 in range(batch_size):
        if last == data.shape[0]:
            last = 0

        mini_batch[i0, :] = data[last, :]
        batch_word2vec[i0, :] = torch.Tensor(word2vecs[labels[last]])
        batch_labels.append(labels[last])

        last += 1

    return mini_batch, batch_word2vec, batch_labels, last

