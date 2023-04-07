import warnings
import numpy as np
import scipy.sparse as sparse
from scipy.special import softmax
from scipy.spatial.distance import cdist
from multiprocessing import Process, Manager
from ellipsoid_node import EllipsoidNode


class EllipsoidClassifier:

    def __init__(self, y_multiple=5, scaling_factor=1):
        # As there might be a lot of distinct labels the set Y can go very large so define a multiple that
        # signifies how many times more points should be in Y compared to X
        self.y_multiple = y_multiple

        # As the data points might be very small in value (as is the case with MEG data) it might be worthwhile to
        # rescale the points to make the optimization easier for the solvers. Rescaling is done as (c * I) * x
        # for every datapoint x, where c is the scaling factor
        self.scaling_factor = scaling_factor

        # Dictionary of nodes of form label: str -> node: Node
        self.nodes = {}

        # Dictionary of form label: str -> vector: np.ndarray for holding the semantic space information
        self.semantic_vectors = {}

        # Helpful constants
        self.data_dim = None
        self.sem_dim = None

    def __repr__(self):
        return f"EllipsoidClassifier(y_multiple={self.y_multiple}, scaling_factor={self.scaling_factor})"

    @staticmethod
    def __group_points__(X, y):
        """
        Helper function used by the training functions that groups the datapoints by their
        labels into a dictionary of form label -> datapoints
        :param X: The data array. A numpy.ndarray of shape (n, d)
        :param y: The label array. A numpy.ndarray of shape (n,)
        :return: Dictionary of form label: str -> points: numpy.ndarray of shape (m, d)
        """
        ret_dict = {}
        # Get the distinct labels
        unique_labels = np.unique(y)

        # Loop over the labels
        for label in unique_labels:
            # Find the indexes at which the label in question is found in array y
            label_indexes = (y == label).nonzero()[0]

            # Get the corresponding datapoints and update the dictionary
            ret_dict[label] = X[label_indexes]

        return ret_dict

    def __train_node__(self, label, grouped_points, model, B, decomposition_splits, omega):
        """
        Helper function that wraps multiple operations within it
        :param label: The label of the node to be trained
        :param grouped_points: The datapoints grouped by label
        str -> points: numpy.ndarray of shape (m, d) containing all points and labels
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
        X = grouped_points[label]

        # Combine the rest of the points into one numpy array
        Y = np.concatenate([value for key, value in grouped_points.items() if key != label])

        # Take a random sample from set Y
        if Y.shape[0] > X.shape[0] * self.y_multiple:
            Y = Y[np.random.randint(Y.shape[0], size=X.shape[0] * self.y_multiple), :]

        # Rescale the datapoints
        if self.scaling_factor != 1:
            scaler = self.scaling_factor * np.ones((X.shape[1],))
            X = np.array([scaler * x for x in X])
            Y = np.array([scaler * y for y in Y])

        # Train the node
        new_node = EllipsoidNode(label)
        new_node.find_center(X)
        new_node.find_matrix(X, Y, model, B, decomposition_splits, omega)
        self.nodes[label] = new_node

    def __par_train_node__(self, label, grouped_points, ret_dict, model, B, decomposition_splits, omega):
        """
        Helper function that wraps multiple operations within it so that they can be called in parallel pool
        :param label: The label of the node to be trained
        :param grouped_points: The datapoints grouped by label
        :param ret_dict: multiprocessing.Manager.dict() to store the return value
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
        X = grouped_points[label]

        # Combine the rest of the points into one numpy array
        Y = np.concatenate([value for key, value in grouped_points.items() if key != label])

        # Take a random sample from set Y
        if Y.shape[0] > X.shape[0] * self.y_multiple:
            Y = Y[np.random.randint(Y.shape[0], size=X.shape[0] * self.y_multiple), :]

        # Rescale the datapoints
        if self.scaling_factor != 1:
            scaler = self.scaling_factor * np.ones((X.shape[1],))
            X = np.array([scaler * x for x in X])
            Y = np.array([scaler * y for y in Y])

        # Train the node
        new_node = EllipsoidNode(label)
        new_node.find_center(X)
        new_node.find_matrix(X, Y, model, B, decomposition_splits, omega)
        ret_dict[label] = new_node

    def train(self, X, y, model, parallel=False, B=None, decomposition_splits=0, omega=1):
        """
        Train the model by sequentially optimizing the balls for each individual node.
        :param X: The data array. A numpy.ndarray of shape (n, d)
        :param y: The label array. A numpy.ndarray of shape (n,)
        :param model: str telling which model is to be used. Currently, supported options are "ind", "full" and
        "banded". If banded is chosen then the set of deviations B must be specified
        :param parallel: Optional. bool telling if the training of the nodes should be parallelized using pythons
        multiprocessing library. Defaults to False
        :param B: Optional. array_like giving the deviations from the main diagonal associated with the band. If None
        the independent model will be used, if "full" the full model will be used and if array_like of integers the
        given band will be used. Do note that all integers in the band must be positive and thus only the triangular
        form of the deviations is needed. Defaults to None
        :param decomposition_splits: Optional. int denoting to how many subproblems the given optimization task should
        be divided. Defaults to 0
        :param omega: Optional. float representing the correcting coefficient for possible disparity between the sizes
        of sets X and Y. Defaults to 1
        :return: Void
        """
        assert X.shape[0] == y.shape[0], "The number of points must match with the number of labels!"
        self.data_dim = X.shape[1]

        # Group datapoints by label
        grouped_points = self.__group_points__(X, y)

        if parallel:
            # Initialize processes
            manager = Manager()
            ret_dict = manager.dict()
            processes = []
            for label in grouped_points:
                process = Process(target=self.__par_train_node__, args=(label, grouped_points, ret_dict, model, B, decomposition_splits, omega))
                processes.append(process)
                process.start()

            # Complete the processes
            for proc in processes:
                proc.join()

            self.nodes = ret_dict

        else:
            for label in grouped_points:
                self.__train_node__(label, grouped_points, model, B, decomposition_splits, omega)

    def eval(self):
        """
        Evaluate the total training accuracy of the model
        :return: The total training accuracy as a floating point number
        """
        assert len(self.nodes) > 0, "The model must be trained!"
        return sum([node.accuracy() for label, node in self.nodes.items()]) / len(self.nodes)

    def add_sematic_vectors(self, S, y):
        """
        Adds the semantic vectors into memory for use in zero-shot predicting
        Note! Each label used in training should be in the label array and each label can be exactly
        once in the label array
        :param S: The semantic data array. A numpy.ndarray of shape (n0, s)
        :param y: The label array. A numpy.ndarray of shape (n0,)
        :return: Void
        """
        assert y.shape[0] == np.unique(y).shape[0], "All labels should be unique!"
        assert S.shape[0] == y.shape[0], "The number of points must match with the number of labels!"
        self.sem_dim = S.shape[1]

        # Check that there is a semantic vector for each of the possibly existing nodes
        if len(self.nodes) != 0:
            for label, node in self.nodes.items():
                if label not in y:
                    raise RuntimeError("\nSemantic vector not provided for all existing labels")

        # Use the given arrays to create a dictionary and store it in memory
        sem_dict = {}
        for i in range(0, y.shape[0]):
            sem_dict[y[i]] = S[i]

        self.semantic_vectors = sem_dict

    def add_matrices_and_centers(self, matrices, centers, labels):
        """
        Function for generating the nodes for already existing matrices and center points
        :param matrices: array_like of either numpy.ndarray or scipy.sparse.csr_matrix objects of shape (dim, dim)
        denoting the characteristic matrices of the ellipsoids
        :param centers: array_like of numpy.ndarray objects of shape (dim, ) denoting the center points of the ellipsoids
        :param labels: array_like of labels corresponding with the matrices and centers
        :return: Void
        """
        assert len(matrices) == len(centers) == len(labels), "The lengths of the inputs must match!"
        assert type(matrices[0]) == sparse.csr_matrix or type(matrices[0]) == np.ndarray, "Invalid type passed for the matrix!"
        self.data_dim = centers[0].shape[0]

        for i in range(len(labels)):
            new_node = EllipsoidNode(labels[i])
            new_node.add_matrix_and_center(matrices[i], centers[i])
            self.nodes[labels[i]] = new_node

    def predict(self, X_test, y_test=None, use_sem=False, metric="euclidean"):
        """
        Function for using the defined model in predicting the labels of passed data points
        :param X_test: numpy.ndarray of shape (n_samples, dim) containing the testing data points
        :param y_test: Optional. numpy.ndarray of shape (n_samples, ) containing the corresponding labels. If passed
        the function will return the found prediction accuracy, otherwise the predictions themselves. Defaults to None
        :param use_sem: Optional. bool telling whether to use semantic vectors in the predictions. Note if no
        semantic vectors have been passed reverts to using the center points. Defaults to False
        :param metric: Optional. str denoting the metric that is used when finding the nearest center/semantic
        vector. Allowed metrics are the ones that scipy has an implementation for. Defaults to "euclidean"
        :return: numpy.ndarray of predictions or float prediction accuracy depending on the parameter choices
        """
        assert len(self.nodes) > 0, "The model must be trained!"
        assert X_test.shape[1] == self.data_dim, "The dimension of the passed datapoints should be the same as the ones used in training!"

        if y_test is not None:
            assert X_test.shape[0] == y_test.shape[0], "The number of points must match with the number of labels!"

        if use_sem and len(self.semantic_vectors) == 0:
            warnings.warn("No semantic vectors passed to the model. Center points used in the predictions instead.", RuntimeWarning)
            use_sem = False

        if use_sem:
            search_dict = self.semantic_vectors
        else:
            search_dict = dict([(label, node.center()) for label, node in self.nodes.items()])

        # Scale the points in X_test to match the ones used in training
        if self.scaling_factor != 1:
            scaler = self.scaling_factor * np.ones((X_test.shape[1],))
            X_test = np.array([scaler * x for x in X_test])

        predictions = []
        for i in range(X_test.shape[0]):
            point = X_test[i]

            # Compute the distances from the point to the surface of each of the balls
            dists = [(label, node.dist(point)) for label, node in self.nodes.items()]
            labels = [tup[0] for tup in dists]
            dists = [tup[1] for tup in dists]

            # Convert the distances to weights
            weights = [max(dists) - dist for dist in dists]
            weights = softmax(weights)

            # Compute the weighted average for the vector to be approximated
            avg = np.zeros((self.sem_dim,))

            for i, label in enumerate(labels):
                avg += weights[i] * search_dict[label]

            # Do a brute force search for the 1-nearest neighbour
            best_dist = float('inf')
            best_label = ""
            for label in self.nodes:
                vect = search_dict[label]
                dist = cdist([avg], [vect], metric)[0][0]
                if dist < best_dist:
                    best_label = label
                    best_dist = dist

            predictions.append(best_label)

        predictions = np.array(predictions)

        if y_test is not None:
            return np.sum(predictions == y_test) / y_test.shape[0]

        return np.ndarray(predictions)
