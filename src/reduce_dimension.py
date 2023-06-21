import numpy as np
from skimage.transform import resize
from sklearn import decomposition


def reduce_dimension_resize(X, dim_tup):
    """
    Function wrapper for calling skimage.transform.resize for each black and white image in X
    :param X: ndarray of shape (n_samples, x_dim, y_dim) containing the images to be resized
    :param dim_tup: tuple of form (new_x_dim, new_y_dim) denoting the new dimensions of the images
    :return: ndarray of shape (n_samples, new_x_dim, new_y_dim) containing the resized images
    """
    X0 = []
    for x in X:
        X0.append(resize(x, dim_tup))

    return np.array(X0)


def reduce_dimension_pca(X, new_dim, scaling_factor=1):
    """
    Function wrapped for sklearn.decomposition.PCA that applies a PCA decomposition to reduce the dimension of each
    datapoint in X
    :param X: ndarray of shape (n_samples, dim) containing the datapoints to be resized
    :param new_dim: int denoting the new dimension
    :param scaling_factor: Optional. int denoting the amount by which the datapoints in X should be scaled.
    Defaults to 1
    :return: ndarray of shape (n_samples, new_dim) containing the resized datapoints
    """
    assert new_dim <= len(X), "PCA can only be applied if the number of datapoints exceeds or equals the new dimension!"

    if scaling_factor != 1:
        X = np.array([scaling_factor * x for x in X])

    pca = decomposition.PCA()
    pca.n_components = new_dim
    pca_X = pca.fit_transform(X)

    return pca_X
