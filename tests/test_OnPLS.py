# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 23:07:26 2016

Copyright (c) 2016, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
from nose.tools import assert_less

import numpy as np

import OnPLS.consts as consts
import OnPLS.estimators as estimators
import OnPLS.utils as utils

import tests


class TestOnPLS(tests.TestCase):

    def test_comparison_to_nPLS(self):
        np.random.seed(42)

        verbose = 1

        n, p_1, p_2, p_3 = 10, 5, 10, 15

        # Generate data:
        t = np.sort(np.random.randn(n, 1), axis=0)
        t = t / np.linalg.norm(t)
        to1 = np.ones((n, 1))
        to1 = to1 - utils.project(to1, t)
        to1 = to1 / np.linalg.norm(to1)
        to2 = np.random.rand(n, 1)
        to2 = to2 - utils.project(to2, t) \
                  - utils.project(to2, to1)
        to2 = to2 / np.linalg.norm(to2)
        to3 = np.random.rand(n, 1)
        to3 = to3 - utils.project(to3, t) \
                  - utils.project(to3, to1) \
                  - utils.project(to3, to2)
        to3 = to3 / np.linalg.norm(to3)
        assert(np.dot(t.T, to1) < 5e-15)
        assert(np.dot(t.T, to2) < 5e-15)
        assert(np.dot(t.T, to3) < 5e-15)
        assert(np.dot(to1.T, to2) < 5e-15)
        assert(np.dot(to1.T, to3) < 5e-15)
        assert(np.dot(to2.T, to3) < 5e-15)
        p1 = np.sort(np.random.randn(p_1, 1), axis=0)
        p2 = np.sort(np.random.randn(p_2, 1), axis=0)
        p3 = np.sort(np.random.randn(p_3, 1), axis=0)
        po1 = np.sort(np.random.randn(p_1, 1), axis=0)
        po2 = np.sort(np.random.randn(p_2, 1), axis=0)
        po3 = np.sort(np.random.randn(p_3, 1), axis=0)

        X1 = np.dot(t, p1.T) + np.dot(to1, po1.T)
        X2 = np.dot(t, p2.T) + np.dot(to2, po2.T)
        X3 = np.dot(t, p3.T) + np.dot(to3, po3.T)
#        Xte1 = np.dot(t, p1.T)
#        Xte2 = np.dot(t, p2.T)
#        Xte3 = np.dot(t, p3.T)

        predComp = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        precomputedW = None

        # OnPLS model:
        orthComp = [1, 1, 1]
        model = None
        onpls = estimators.OnPLS(predComp, orthComp, model, precomputedW,
                                 numReps=1, verbose=verbose)
        onpls.fit([X1, X2, X3])

        Xhat, That = onpls.predict([X1, X2, X3], [2], return_scores=True)
        onpls_score = onpls.score([X1, X2, X3])
        assert(onpls_score > 0.999)

#        Xhat, That = onpls.predict([Xte1, Xte2, Xte3], [2],
#                                   return_scores=True)

        if np.dot(t.T, That[0]) < 0.0:
            That[0] = -That[0]
#        print np.linalg.norm(Xhat[0] - np.dot(t, p3.T))
        assert(np.linalg.norm(Xhat[0] - np.dot(t, p3.T)) < 5e-13)

        # nPLS model:
        npls = estimators.nPLS(predComp, precomputedA=None, numReps=1,
                               randomState=None, verbose=verbose)
        npls.fit([X1, X2, X3])

        Xhat, That = npls.predict([X1, X2, X3], [2], return_scores=True)
#        Xhat, That = npls.predict([Xte1, Xte2, Xte3], [2],
#                                  return_scores=True)

        if np.dot(t.T, That[0]) < 0.0:
            That[0] = -That[0]
#        print np.linalg.norm(Xhat[0] - np.dot(t, p3.T))
        assert(np.linalg.norm(Xhat[0] - np.dot(t, p3.T)) < 1.5)
        npls_score = npls.score([X1, X2, X3])
#        print abs(npls_score - 0.37736)
        assert(abs(npls_score - 0.37736) < 5e-6)

        assert(onpls_score > npls_score)

    def test_comparison_to_nPLS_2comps(self):
        np.random.seed(42)

        verbose = 1

        n, p_1, p_2, p_3 = 10, 5, 10, 15

        # Generate data:
        t1 = np.sort(np.random.randn(n, 1), axis=0)
        t1 = t1 / np.linalg.norm(t1)

        t2 = np.random.randn(n, 1)
        t2 = t2 - utils.project(t2, t1)
        t2 = t2 / np.linalg.norm(t2)

        to1 = np.ones((n, 1))
        to1 = to1 - utils.project(to1, t1) \
                  - utils.project(to1, t2)
        to1 = to1 / np.linalg.norm(to1)

        to2 = np.random.rand(n, 1)
        to2 = to2 - utils.project(to2, t1) \
                  - utils.project(to2, t2) \
                  - utils.project(to2, to1)
        to2 = to2 / np.linalg.norm(to2)

        to3 = np.random.rand(n, 1)
        to3 = to3 - utils.project(to3, t1) \
                  - utils.project(to3, t2) \
                  - utils.project(to3, to1) \
                  - utils.project(to3, to2)
        to3 = to3 / np.linalg.norm(to3)

        assert(np.abs(np.dot(t1.T, t2)) < 5e-15)
        assert(np.abs(np.dot(t1.T, to1)) < 5e-15)
        assert(np.abs(np.dot(t1.T, to2)) < 5e-15)
        assert(np.abs(np.dot(t1.T, to3)) < 5e-15)
        assert(np.abs(np.dot(t2.T, to1)) < 5e-15)
        assert(np.abs(np.dot(t2.T, to2)) < 5e-15)
        assert(np.abs(np.dot(t2.T, to3)) < 5e-15)
        assert(np.abs(np.dot(to1.T, to2)) < 5e-15)
        assert(np.abs(np.dot(to1.T, to3)) < 5e-15)
        assert(np.abs(np.dot(to2.T, to3)) < 5e-15)

        p11 = np.sort(np.random.randn(p_1, 1), axis=0)
        p12 = np.sort(np.random.randn(p_2, 1), axis=0)
        p13 = np.sort(np.random.randn(p_3, 1), axis=0)
        p21 = np.sort(np.random.randn(p_1, 1), axis=0)
        p22 = np.sort(np.random.randn(p_2, 1), axis=0)
        p23 = np.sort(np.random.randn(p_3, 1), axis=0)
        po1 = np.sort(np.random.randn(p_1, 1), axis=0)
        po2 = np.sort(np.random.randn(p_2, 1), axis=0)
        po3 = np.sort(np.random.randn(p_3, 1), axis=0)

        X1 = np.dot(t1, p11.T) + np.dot(t2, p21.T) + np.dot(to1, po1.T)
        X2 = np.dot(t1, p12.T) + np.dot(t2, p22.T) + np.dot(to2, po2.T)
        X3 = np.dot(t1, p13.T) + np.dot(t2, p23.T) + np.dot(to3, po3.T)
#        Xte1 = np.dot(t, p1.T)
#        Xte2 = np.dot(t, p2.T)
#        Xte3 = np.dot(t, p3.T)

        predComp = [[0, 2, 2], [2, 0, 2], [2, 2, 0]]
        precomputedW = None

        # OnPLS model:
        orthComp = [1, 1, 1]
        model = None
        onpls = estimators.OnPLS(predComp, orthComp, model, precomputedW,
                                 numReps=1, verbose=verbose)
        onpls.fit([X1, X2, X3])

        Xhat, That = onpls.predict([X1, X2, X3], [2], return_scores=True)
        onpls_score = onpls.score([X1, X2, X3])
        assert(onpls_score > 0.85)

#        Xhat, That = onpls.predict([Xte1, Xte2, Xte3], [2],
#                                   return_scores=True)

        if np.dot(t1.T, That[0]) < 0.0:
            That[0] = -That[0]
#        print np.linalg.norm(Xhat[0] - np.dot(t, p3.T))
#        assert(np.linalg.norm(Xhat[0] - np.dot(t1, p13.T)) < 5e-13)

        # nPLS model:
        npls = estimators.nPLS(predComp, precomputedA=None, numReps=1,
                               randomState=None, verbose=verbose)
        npls.fit([X1, X2, X3])

        Xhat, That = npls.predict([X1, X2, X3], [2], return_scores=True)
#        Xhat, That = npls.predict([Xte1, Xte2, Xte3], [2],
#                                  return_scores=True)

        if np.dot(t1.T, That[0]) < 0.0:
            That[0] = -That[0]
#        print np.linalg.norm(Xhat[0] - np.dot(t, p3.T))
        assert(np.linalg.norm(Xhat[0] - np.dot(t1, p13.T)) < 3.3)
        npls_score = npls.score([X1, X2, X3])
#        print abs(npls_score - 0.37736)
        assert(abs(npls_score - 0.394976) < 5e-6)

        assert(onpls_score > npls_score)


if __name__ == "__main__":
    import unittest
    unittest.main()
