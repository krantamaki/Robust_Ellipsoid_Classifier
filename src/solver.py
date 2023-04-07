import numpy as np
import scipy.sparse as sparse
import pyomo.environ as pyo


def to_index_dict(A):
    """
    Function that converts a given numpy.ndarray to a python dictionary where the key is the tuple of indexes for the
    elements of the ndarray. Additionally, shifts the indexes by one so that they start at 1 instead of 0
    :param A: numpy.ndarray of shape (n, m) or (n, )
    :return: python dictionary of form (indexes) -> value
    """

    if len(A.shape) == 1:
        return dict([(i + 1, val) for i, val in enumerate(A.tolist())])

    assert len(A.shape) == 2, "More than two dimensions not yet supported!"

    d = A.shape[1]
    return dict([((i0 // d + 1, i0 % d + 1), val) for i0, val in enumerate(A.flatten().tolist())])


def dict_to_numpy(A):
    """
    Function that converts a dictionary of form (indexes) -> value to the corresponding numpy array.
    :param A: python dictionary of form (indexes) -> value
    :return: numpy.ndarray of shape (n, m) or (n, )
    """
    keys = list(A.keys())
    values = list(A.values())

    if type(keys[0]) == int:
        return np.array(values)

    assert len(keys[0]) == 2, "More than two dimensions not yet supported!"

    d = max(keys, key=lambda tup: tup[1])[1]
    n = len(values) // d

    return np.array(values).reshape((n, d))


def full_model(X, Y, omega):
    """
    Solves the optimization problem associated with the full model of the classifier. See section
    5.1 in "Zero_shot_classifier_based_on_robust_ellipsoid_optimization.pdf" for more information
    :param X: numpy.ndarray of shape (x_samples, dim) containing the datapoints with the correct label
    :param Y: numpy.ndarray of shape (y_samples, dim) containing the datapoints with the incorrect label
    :param omega: float representing the correcting coefficient for possible disparity between the sizes
    of sets X and Y
    :return: numpy.ndarray corresponding with the found characteristic matrix
    """

    # Define the Pyomo model
    model = pyo.AbstractModel()

    model.m = pyo.Param(within=pyo.NonNegativeIntegers)           # Number of points in set X
    model.n = pyo.Param(within=pyo.NonNegativeIntegers)           # Number of points in set Y
    model.d = pyo.Param(within=pyo.NonNegativeIntegers)           # Dimension of the datapoints
    model.o = pyo.Param()                                         # Omega coefficient

    model.X_i = pyo.RangeSet(1, model.m)                          # Set of indexes for points in set X
    model.Y_i = pyo.RangeSet(1, model.n)                          # Set of indexes for points in set Y
    model.D = pyo.RangeSet(1, model.d)                            # Set of indexes for each feature

    model.X = pyo.Param(model.X_i, model.D)                       # The set X of points with correct label
    model.Y = pyo.Param(model.Y_i, model.D)                       # The set Y of points with incorrect label
    model.c = pyo.Param(model.D)                                  # The center point of the ellipsoid

    model.u = pyo.Var(model.X_i, domain=pyo.NonNegativeReals)     # Relaxing variables for the points in X
    model.v = pyo.Var(model.Y_i, domain=pyo.NonNegativeReals)     # Relaxing variables for the points in X
    model.diag = pyo.Var(model.D, domain=pyo.NonNegativeReals)    # Variables for the diagonal of the matrix
    model.triag = pyo.Var(model.D, model.D)                       # Variables for the off-diagonal elements of the matrix

    # The objective function
    def obj(m):
        return m.o * sum(m.u[i] for i in m.X_i) + sum(m.v[i] for i in m.Y_i)
    model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    # The constraints for points in X
    def X_const(m, k):
        return m.u[k] - sum(m.diag[i] * (m.X[k, i] - m.c[i]) ** 2 for i in m.D) \
               - 2 * sum(sum(m.triag[i, j] * (m.X[k, i] - m.c[i]) * (m.X[k, j] - m.c[j]) for j in pyo.RangeSet(1, i - 1))
                         for i in pyo.RangeSet(2, m.d)) >= 0
    model.X_const = pyo.Constraint(model.X_i, rule=X_const)

    # The constraints for point in Y
    def Y_const(m, k):
        return m.v[k] + sum(m.diag[j] * (m.Y[k, j] - m.c[j]) ** 2 for j in m.D) \
               + 2 * sum(sum(m.triag[i, j] * (m.Y[k, i] - m.c[i]) * (m.Y[k, j] - m.c[j]) for j in pyo.RangeSet(1, i - 1))
                         for i in pyo.RangeSet(2, m.d)) - 2 >= 0
    model.Y_const = pyo.Constraint(model.Y_i, rule=Y_const)

    # Initialize the data passed as parameters to the solver
    data_init = {None: {
        'm': {None: X.shape[0]},
        'n': {None: Y.shape[0]},
        'd': {None: Y.shape[1]},
        'o': {None: omega},
        'X': to_index_dict(X),
        'Y': to_index_dict(Y),
        'c': to_index_dict(np.mean(X, axis=0))
    }}
    instance = model.create_instance(data_init)

    # Define the solver and pass it the created instance
    opt = pyo.SolverFactory("ipopt")
    result = opt.solve(instance)

    diag = dict_to_numpy(instance.diag.extract_values())
    triag = np.tril(dict_to_numpy(instance.triag.extract_values()), -1)  # np.tril applied just to make sure no odd values end up in result

    ret = triag + triag.T + diag

    return ret


def ind_model(X, Y, omega):
    """
    Solves the optimization problem associated with the independent model of the classifier. See section
    5.2 in "Zero_shot_classifier_based_on_robust_ellipsoid_optimization.pdf" for more information
    :param X: numpy.ndarray of shape (x_samples, dim) containing the datapoints with the correct label
    :param Y: numpy.ndarray of shape (y_samples, dim) containing the datapoints with the incorrect label
    :param omega: float representing the correcting coefficient for possible disparity between the sizes
    of sets X and Y
    :return: scipy.sparse.csr matrix corresponding with the found characteristic matrix
    """

    # Define the Pyomo model
    model = pyo.AbstractModel()

    model.m = pyo.Param(within=pyo.NonNegativeIntegers)           # Number of points in set X
    model.n = pyo.Param(within=pyo.NonNegativeIntegers)           # Number of points in set Y
    model.d = pyo.Param(within=pyo.NonNegativeIntegers)           # Dimension of the datapoints
    model.o = pyo.Param()                                         # Omega coefficient

    model.X_i = pyo.RangeSet(1, model.m)                          # Set of indexes for points in set X
    model.Y_i = pyo.RangeSet(1, model.n)                          # Set of indexes for points in set Y
    model.D = pyo.RangeSet(1, model.d)                            # Set of indexes for each feature

    model.X = pyo.Param(model.X_i, model.D)                       # The set X of points with correct label
    model.Y = pyo.Param(model.Y_i, model.D)                       # The set Y of points with incorrect label
    model.c = pyo.Param(model.D)                                  # The center point of the ellipsoid

    model.u = pyo.Var(model.X_i, domain=pyo.NonNegativeReals)     # Relaxing variables for the points in X
    model.v = pyo.Var(model.Y_i, domain=pyo.NonNegativeReals)     # Relaxing variables for the points in X
    model.diag = pyo.Var(model.D, domain=pyo.NonNegativeReals)    # Variables for the diagonal of the matrix

    # The objective function
    def obj(m):
        return m.o * sum(m.u[i] for i in m.X_i) + sum(m.v[i] for i in m.Y_i)
    model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    # The constraints for points in X
    def X_const(m, i):
        return m.u[i] - sum(m.diag[j] * (m.X[i, j] - m.c[j]) ** 2 for j in m.D) >= 0
    model.X_const = pyo.Constraint(model.X_i, rule=X_const)

    # The constraints for point in Y
    def Y_const(m, i):
        return m.v[i] + sum(m.diag[j] * (m.Y[i, j] - m.c[j]) ** 2 for j in m.D) - 2 >= 0
    model.Y_const = pyo.Constraint(model.Y_i, rule=Y_const)

    # Initialize the data passed as parameters to the solver
    data_init = {None: {
        'm': {None: X.shape[0]},
        'n': {None: Y.shape[0]},
        'd': {None: Y.shape[1]},
        'o': {None: omega},
        'X': to_index_dict(X),
        'Y': to_index_dict(Y),
        'c': to_index_dict(np.mean(X, axis=0))
    }}
    instance = model.create_instance(data_init)

    # Define the solver and pass it the created instance
    opt = pyo.SolverFactory("ipopt")
    result = opt.solve(instance)

    # Return a sparse matrix corresponding with the characteristic matrix
    ret = sparse.csr_matrix((X.shape[1], X.shape[1]))
    ret.setdiag(dict_to_numpy(instance.diag.extract_values()))

    return ret


def banded_model(X, Y, omega, B):
    """
    Solves the optimization problem associated with the banded model of the classifier. See section
    5.3 in "Zero_shot_classifier_based_on_robust_ellipsoid_optimization.pdf" for more information
    :param X: numpy.ndarray of shape (x_samples, dim) containing the datapoints with the correct label
    :param Y: numpy.ndarray of shape (y_samples, dim) containing the datapoints with the incorrect label
    :param omega: float representing the correcting coefficient for possible disparity between the sizes
    of sets X and Y
    :param B: numpy.ndarray of shape (m, ) containing the deviations from the main diagonal associated with the band
    :return: scipy.sparse.csr matrix corresponding with the found characteristic matrix
    """

    # Define the Pyomo model
    model = pyo.AbstractModel()

    model.m = pyo.Param(within=pyo.NonNegativeIntegers)           # Number of points in set X
    model.n = pyo.Param(within=pyo.NonNegativeIntegers)           # Number of points in set Y
    model.d = pyo.Param(within=pyo.NonNegativeIntegers)           # Dimension of the datapoints
    model.o = pyo.Param()                                         # Omega coefficient
    model.lenB = pyo.Param(within=pyo.NonNegativeIntegers)        # Number of deviations in B

    model.X_i = pyo.RangeSet(1, model.m)                          # Set of indexes for points in set X
    model.Y_i = pyo.RangeSet(1, model.n)                          # Set of indexes for points in set Y
    model.D = pyo.RangeSet(1, model.d)                            # Set of indexes for each feature
    model.B_i = pyo.RangeSet(1, model.lenB)                       # Set of indexes for the deviations

    model.B = pyo.Param(model.B_i)                                # Set of deviations from the main diagonal
    model.X = pyo.Param(model.X_i, model.D)                       # The set X of points with correct label
    model.Y = pyo.Param(model.Y_i, model.D)                       # The set Y of points with incorrect label
    model.c = pyo.Param(model.D)                                  # The center point of the ellipsoid

    model.u = pyo.Var(model.X_i, domain=pyo.NonNegativeReals)     # Relaxing variables for the points in X
    model.v = pyo.Var(model.Y_i, domain=pyo.NonNegativeReals)     # Relaxing variables for the points in X
    model.diag = pyo.Var(model.D, domain=pyo.NonNegativeReals)    # Variables for the diagonal of the matrix
    model.band = pyo.Var(model.B, model.D)                        # Variables for the band of the matrix

    # The objective function
    def obj(m):
        return m.o * sum(m.u[i] for i in m.X_i) + sum(m.v[i] for i in m.Y_i)
    model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    # The constraints for points in X
    def X_const(m, k):
        return m.u[k] - sum(m.diag[i] * (m.X[k, i] - m.c[i]) ** 2 for i in m.D) \
               - 2 * sum(sum(m.band[j, i - m.B[j]] * (m.X[k, i] - m.c[i]) * (m.X[k, i - m.B[j]] - m.c[i - m.B[j]])
                             for i in pyo.RangeSet(m.B[j] + 1, m.d)) for j in m.B_i) >= 0
    model.X_const = pyo.Constraint(model.X_i, rule=X_const)

    # The constraints for point in Y
    def Y_const(m, k):
        return m.v[k] + sum(m.diag[i] * (m.Y[k, i] - m.c[i]) ** 2 for i in m.D) \
               - 2 * sum(sum(m.band[j, i - m.B[j]] * (m.Y[k, i] - m.c[i]) * (m.Y[k, i - m.B[j]] - m.c[i - m.B[j]])
                             for i in pyo.RangeSet(m.B[j] + 1, m.d)) for j in m.B_i) - 2 >= 0
    model.Y_const = pyo.Constraint(model.Y_i, rule=Y_const)

    # Initialize the data passed as parameters to the solver
    data_init = {None: {
        'm': {None: X.shape[0]},
        'n': {None: Y.shape[0]},
        'd': {None: Y.shape[1]},
        'o': {None: omega},
        'lenB': {None, len(B)},
        'B': {None: B.tolist()},
        'X': to_index_dict(X),
        'Y': to_index_dict(Y),
        'c': to_index_dict(np.mean(X, axis=0))
    }}
    instance = model.create_instance(data_init)

    # Define the solver and pass it the created instance
    opt = pyo.SolverFactory("ipopt")
    result = opt.solve(instance)

    # Return a sparse matrix corresponding with the characteristic matrix
    ret = sparse.csr_matrix((X.shape[1], X.shape[1]))
    ret.setdiag(dict_to_numpy(instance.diag.extract_values()))

    band = dict_to_numpy(instance.band.extract_values())
    for i, b in enumerate(B.tolist()):
        ret.setdiag(band[i, :], b)
        ret.setdiag(band[i, :], -b)

    return ret


def decomposition(X, Y, model, B, decomposition_splits, omega):
    """
    Decomposes the problem into wanted sized chunks as laid out in section 5.4 of
    "Zero_shot_classifier_based_on_robust_ellipsoid_optimization.pdf" and calls the wanted model for
    each subproblem
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
    """
    # Empty list for the found matrices
    matrices = []

    # The number of dimensions in each subproblem
    d0 = X.shape[1] // decomposition_splits

    for i in range(decomposition_splits):

        # Split the datapoints to the needed subspace
        X0 = X[:, i * d0:(i + 1) * d0]
        Y0 = Y[:, i * d0:(i + 1) * d0]

        if model == "ind":
            matrices.append(ind_model(X0, Y0, omega))
        elif model == "banded":
            matrices.append(banded_model(X0, Y0, omega, B))
        else:
            raise RuntimeError("Improper model definition given!")

    return sparse.block_diag(matrices, format="csr")

