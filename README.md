OnPLS
=====

OnPLS: Orthogonal Projections to Latent Structures in Multiblock and Path Model Data Analysis

OnPLS is a Python package for multiblock data analysis with prefiltering of unique and locally joint variation.


Installation
------------

The reference environment for OnPLS is Ubuntu 14.04 LTS with Python 2.7.6 or Python 3.4.3 and Numpy 1.8.2.

Unless you already have Numpy installed, you need to install it:
```
$ sudo apt-get install python-numpy
```
or
```
$ sudo apt-get install python3-numpy
```

In order to run the tests, you may also need to install Nose:
```
$ sudo apt-get install python-nose
```
or
```
$ sudo apt-get install python3-nose
```

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

Stable reseases with setup scripts will be included in future versions.

You are now ready to use your fresh installation of OnPLS!


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
```
