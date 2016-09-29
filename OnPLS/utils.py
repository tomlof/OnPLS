# -*- coding: utf-8 -*-
"""
The :mod:`OnPLS.utils` module contains useful functions and functionality used
throughout the package.

Created on Fri Jul 29 22:10:29 2016

Copyright (c) 2016, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import numpy as np

try:
    from . import consts  # When imported as a package.
except ValueError:
    import OnPLS.consts as consts  # When run as a program.

__all__ = ["leastNonZero", "normaliseColumns", "sumOfSquares",
           "sumOfCovariances", "cov", "project"]


def leastNonZero(A):

    A = np.array(A)
    return np.min(A[A > 0.0])


def normaliseColumns(A):

    norms = np.linalg.norm(A, axis=0)
    # Avoid division by zero: Define 0 / 0 = 0.
    norms[np.abs(norms) < consts.TOLERANCE] = 1.0
    A = A / norms

    return A


def sumOfSquares(A):

    return np.sum(A**2.0)


def sumOfCovariances(T, predComp, comp=None):

    n = len(T)
    sumCov = [0] * n

    if comp is None:
        comps = range(T[0].shape[1])
    else:
        comps = comp

    for c in comps:
        for i in range(n):
            Ti = T[i]
            for j in range(n):
                Tj = T[j]
                if predComp[i][j] > 0:
                    sumCov[i] = sumCov[i] + cov(Ti[:, c], Tj[:, c])

    return sumCov


def cov(x, y):
    n = np.max(x.shape)
    x_ = x - np.mean(x)
    y_ = y - np.mean(y)

    coef = np.dot(x_.T, y_) / n

    return coef


def project(v, u):
    """ Project v onto u.
    """
    return (np.dot(v.T, u) / np.dot(u.T, u)) * u
