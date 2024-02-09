OnPLS
=====

OnPLS: Orthogonal Projections to Latent Structures in Multiblock and Path Model
Data Analysis.

OnPLS is a Python package for multiblock data analysis with prefiltering of
unique and locally joint variation.


Installation
------------

The reference environment for OnPLS is Ubuntu 20.04 LTS with Python 3.10 and
Numpy 1.26. It was originally made to also run with Python 2.7, but has not
recently been tested with Python 2.7.

We recommend that you use a Python package manager, such as Anaconda or pip.

The main requirement is Numpy. To run the tests, you also need Nose.


**Downloading the latest development version**

Clone the Github repository

```
$ git clone https://github.com/tomlof/OnPLS.git
```
Preferably, you would fork it first and clone your own repository.

Add OnPLS to your Python path:
```
$ export $PYTHONPATH=$PYTHONPATH:/directory/to/OnPLS
```

Stable reseases with setup scripts may be included in future versions.

You are now ready to use your fresh installation of OnPLS!

Contributions are very welcome!


Quick start
-----------

A simple example of the usage:

```python
import numpy as np
import OnPLS

np.random.seed(42)

n, p_1, p_2, p_3 = 4, 3, 4, 5
t = np.sort(np.random.randn(n, 1), axis=0)
p1 = np.sort(np.random.randn(p_1, 1), axis=0)
p2 = np.sort(np.random.randn(p_2, 1), axis=0)
p3 = np.sort(np.random.randn(p_3, 1), axis=0)
X1 = np.dot(t, p1.T) + 0.1 * np.random.randn(n, p_1)
X2 = np.dot(t, p2.T) + 0.1 * np.random.randn(n, p_2)
X3 = np.dot(t, p3.T) + 0.1 * np.random.randn(n, p_3)

# Define the connections between blocks
predComp = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
# Define the numbers of non-global components
orthComp = [1, 1, 1]

# Create the estimator
onpls = OnPLS.estimators.OnPLS(predComp, orthComp)

# Fit a model
onpls.fit([X1, X2, X3])

# Perform prediction of all matrices from all connected matrices
Xhat = onpls.predict([X1, X2, X3])

# Compute prediction score
score = onpls.score([X1, X2, X3])

cv_scores = OnPLS.resampling.cross_validation(onpls, [X1, X2, X3], cv_rounds=4)
```
