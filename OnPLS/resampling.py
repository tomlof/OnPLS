# -*- coding: utf-8 -*-
"""
The :mod:`OnPLS.resampling` module contains resampling methods that can be used
to determine statistics from an estimator.

Created on Wed Nov 23 21:02:03 2016

Copyright (c) 2016, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import numpy as np

import OnPLS.estimators as estimators

__all__ = ["cross_validation"]


def cross_validation(estimator, X, cv_rounds=7, random_state=None):

    if isinstance(estimator, estimators.BaseUniblock):
        if isinstance(X, np.ndarray):
            X = [X]

    n = len(X)

    N = X[0].shape[0]

    cv_rounds = min(max(1, N), int(cv_rounds))

    scores = []
    for k in range(cv_rounds):

        test_samples = list(range(k, N, cv_rounds))
        train_samples = list(set(range(0, N)).difference(test_samples))

        Xtest = [0] * n
        Xtrain = [0] * n
        for i in range(n):
            Xi = X[i]
            Xtest[i] = Xi[test_samples, :]
            Xtrain[i] = Xi[train_samples, :]

        estimator.fit(Xtrain)

        score = estimator.score(Xtest)
        scores.append(score)

    return scores
