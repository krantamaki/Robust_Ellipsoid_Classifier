import numpy as np
import scipy.sparse as sparse
from solver import *


class EllipsoidNode:

    def __init__(self, label):
        # The label of the node in question
        self.label = label

        # Define private variables
        self.__center = None         # Center point of the ellipse. When undefined equals None
        self.__matrix = None         # Characteristic matrix of the ellipse. When undefined equals None
        self.__acc = None            # The final training accuracy. When undefined equals None

    def __str__(self):
        if self.__matrix is None:
            return f"Undefined ellipsoid associated with label {self.label}"
        else:
            return f"Defined ellipsoid associated with label {self.label}"

    def __repr__(self):
        return f"EllipseNode({self.label})"

    def __hash__(self):
        return hash(self.label)

    # Define functions for accessing the private variables
    def center(self):
        return self.__center

    def matrix(self):
        return self.__matrix

    def accuracy(self):
        return self.__acc

    def add_matrix_and_center(self, matrix, center):
        """
        Save some matrix and center point computed elsewhere as the center and characteristic matrix of this ellipsoid
        :param matrix: Either numpy.ndarray or scipy.sparse.csr_matrix of shape (dim, dim) denoting the characteristic
        matrix of the ellipsoid
        :param center: numpy.ndarray of shape (dim, ) denoting the center point of the ellipsoid
        :return: Void
        """
        self.__center = center
        self.__matrix = matrix

    def find_center(self, X):
        """
        Find the center point of the sphere. This is just the arithmetic mean of the data points
        :param X: numpy.ndarray of shape (m, d) containing the data points with the correct label
        :return: Void
        """
        self.__center = np.mean(X, axis=0)

    def find_matrix(self, X, Y, model, B, decomposition_splits, omega):
        """
        Function wrapper for the functions in solver.py that call the Ipopt solver to solve for the
        characteristic matrix A

        That is solve the constrained nonlinear optimization problem

            min. sum(u_i for i in 1, 2, ... , |X|) + sum(v_i for i in 1, 2, ... , |Y|)
            s.t. u_i - (x_i - c)^T A(x_i - c) >= 0      for i in 1, 2, ... , |X|
                 v_i + (y_i - c)^T A(y_i - c) - 2 >= 0  for i in 1, 2, ... , |Y|
                 u_i >= 0                               for i in 1, 2, ... , |X|
                 v_i >= 0                               for i in 1, 2, ... , |Y|

            where u_i and v_i are variables created from the relaxation of the original condition and
            c is the center point of the ellipsoid

        NOTE! The form of A will depend on the chosen model. To learn more please see the theoretical work in
        "Zero_shot_classifier_based_on_robust_ellipsoid_optimization.pdf"

        :param X: numpy.ndarray of shape (x_samples, dim) containing the datapoints with the correct label
        :param Y: numpy.ndarray of shape (y_samples, dim) containing the datapoints with the incorrect label
        :param model: str telling which model is to be used. Currently, supported options are "ind", "full" and
        "banded". If banded is chosen then the set of deviations B must be specified
        :param B: array_like giving the deviations from the main diagonal associated with the band. If None
        the independent model will be used, if "full" the full model will be used and if array_like of integers the
        given band will be used. Do note that all integers in the band must be positive and thus only the triangular
        form of the deviations is needed
        :param decomposition_splits: int denoting to how many subproblems the given optimization task should
        be divided
        :param omega: float representing the correcting coefficient for possible disparity between the sizes
        of sets X and Y
        :return: Void
        """
        assert X.shape[1] == Y.shape[1], "The dimensions of the datapoints must match!"
        assert omega > 0, "The given coefficient omega must be positive!"
        assert decomposition_splits >= 0, "The number splits must be non-negative!"

        if decomposition_splits != 0:
            assert X.shape[1] % decomposition_splits == 0, "Can't divide the problem into given amount of equal subproblems!"

        if model.lower().strip() == "full":
            assert decomposition_splits == 0, "Decomposition not supported for full matrices!"
            self.__matrix = full_model(X, Y, omega)

        elif model.lower().strip() == "ind":
            if decomposition_splits != 0:
                self.__matrix = decomposition(X, Y, model, B, decomposition_splits, omega)
            else:
                self.__matrix = ind_model(X, Y, omega)

        elif model.lower().strip() == "banded" and B is not None:
            if decomposition_splits != 0:
                self.__matrix = decomposition(X, Y, model, B, decomposition_splits, omega)
            else:
                self.__matrix = banded_model(X, Y, omega, B)

        else:
            raise RuntimeError("Improper model definition given!")

    def dist(self, point):
        """
        Computes the distance from the surface of the ellipse to the given point as (p - c)^T A(p - c)
        :param point: ndarray of shape (dim, ) denoting the point to which the distance is computed
        :return: The computed distance as float
        """
        assert self.__center is not None and self.__matrix is not None, "Distance can be computed only if the center and the characteristic matrix are defined!"

        if type(self.__matrix) == sparse.csr_matrix:
            return np.dot((point - self.__center), self.__matrix @ (point - self.__center).T)

        return np.dot((point - self.__center), np.matmul(self.__matrix, (point - self.__center)))
