# Robust Ellipsoid Classifier
This repository contains the source code (in directory src) and a few examples 
(in examples directory) for a SVM based robust ellipsoid classifier that is also
capable of zero-shot learning. The theoretical outline for how this classifier
works is laid out in "Zero_shot_classifier_based_on_robust_ellipsoid_optimization.pdf"
file. Do note that it is a work in progress and might contain some typos and currently
lacks practically all references and proofs for theorems.
## Dependencies
The classifier utilises the common scientific computing libraries like numpy and scipy
and crucially the optimization library Pyomo (install with "pip install 'pyomo[optional]'")
and solver Ipopt, which can be downloaded from https://github.com/coin-or/Ipopt/releases.
This site contains the precompiled binaries for Ipopt as building from source can be a bit 
challenging. Once the executable is saved into some directory one must make sure the directory
is found in PATH. It is good to note that the precompiled binary comes with MUMPS linear solver
for solving the linear subproblems, but for very high dimensional classification tasks
this might not be performant enough. In these cases it might be worthwhile to use the linear solvers
from the Hartwell Subroutine Library (https://www.hsl.rl.ac.uk/ipopt/), but these are generally
parallelized and thus require some level of thought from the user. For more information about Pyomo
in general please see the documentation at https://pyomo.readthedocs.io/en/stable/index.html.
Additionally, the source code is written with Python3 in mind and some slight changes might
need to be made if using Python2.
## How to Use
The classifier is meant to be as easy to use as possible and thus has a very similar syntax
to the classifiers implemented in scikit-learn library. Namely, the classifier is a class
(EllipsoidClassifier) that has primary functions .train(X, y, model) and .predict(X), which
behave as one would expect. For reference on the additional functions and parameters
please see the docstrings included in the source code (in file src/ellipsoid_classifier.py).