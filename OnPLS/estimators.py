# -*- coding: utf-8 -*-
"""
The :mod:`OnPLS.estimators` module contains all the statistical methods that
can be applied to data. This module encapsulates all algorithms and allows them
to be applied to data.

Created on Fri Jul 22 21:25:34 2016

Copyright (c) 2016, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
import warnings
from six import with_metaclass

import numpy as np

try:
    from . import utils  # When imported as a package.
except ValueError:
    import OnPLS.utils as utils  # When run as a program.
try:
    from . import consts  # When imported as a package.
except ValueError:
    import OnPLS.consts as consts  # When run as a program.

__all__ = ["BaseEstimator", "BaseUniblock", "BaseTwoblock", "BaseMultiblock",

           "PCA", "nPLS", "OnPLS"]


class BaseEstimator(with_metaclass(abc.ABCMeta, object)):
    """Base class for estimators.

    Parameters
    ----------
    verbose : int, optional
        The verbosity level. Level 0 means print nothing. Level 1 means print
        errors. Level 2 means print errors and warnings. Level 3 means print
        errors, warnings and messages. Default is 2, print errors and warnings.

    eps : float
        Must be positive. The precision used to stop the algorithm. Default is
        consts.TOLERANCE.

    max_iter : int
        Must be positive. The maximum number of iterations to use in the
        algorithm.
    """
    def __init__(self, verbose=2,
                 eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):

        self.verbose = max(0, min(int(verbose), 3))
        self.eps = max(consts.FLOAT_EPSILON, float(eps))
        self.max_iter = max(0, int(max_iter))

        self.nfo = dict()

    @abc.abstractmethod
    def reset(self):
        """Resets the estimator such that it is as if just created.
        """
        raise NotImplementedError('Abstract method "reset" must be '
                                  'specialised!')

    @abc.abstractmethod
    def fit(self, X):
        """Fit the estimator to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

    @abc.abstractmethod
    def predict(self, X):
        """Perform prediction using the fitted parameters.
        """
        raise NotImplementedError('Abstract method "predict" must be '
                                  'specialised!')

    @abc.abstractmethod
    def score(self, X, y):
        """Returns a quality measure of the estimators fitted model.

        What is returned depends on the estimator. See the estimator's
        documentation.
        """
        raise NotImplementedError('Abstract method "score" must be '
                                  'specialised!')

    @abc.abstractmethod
    def parameters(self):
        """Returns a dictionary with the estimator's fitted parameters, e.g.
        regression coefficients.

        What is returned depends on the estimator. See the estimator's
        documentation.
        """
        raise NotImplementedError('Abstract method "parameters" must be '
                                  'specialised!')

    def info(self):
        """Returns information about the estimators run, such as e.g. the time
        of each iteration and the associated objective function value.
        """
        return self.nfo

    def output(self, *strs):
        if self.verbose >= 3:
            outp = ""
            for s in strs:
                outp = outp + str(s)
            print(outp)

    def warn(self, *strs):
        if self.verbose >= 2:
            category = None
            n = len(strs)
            if len(strs) > 0:
                if isinstance(strs[n - 1], Warning):
                    category = strs[n - 1]
                    n -= 1

            warning = ""
            for i in range(n):
                warning = warning + str(strs[i])

            if category is not None:
                warnings.warn(warning, category)
            else:
                warnings.warn(warning)


class BaseUniblock(object):
    pass


class BaseTwoblock(object):
    pass


class BaseMultiblock(object):
    pass


class PCA(BaseUniblock, BaseEstimator):
    """A NIPALS implementation of principal components analysis.

    Parameters
    ----------
    K : int
        The number of components.

    verbose : int, optional
        The verbosity level. Level 0 means print nothing. Level 1 means print
        errors. Level 2 means print errors and warnings. Level 3 means print
        errors, warnings and messages. Default is 2, print errors and warnings.

    eps : float
        Must be positive. The precision used to stop the algorithm. Default is
        consts.TOLERANCE.

    max_iter : int
        Must be positive. The maximum number of iterations to use in the
        algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> import OnPLS.estimators as estimators
    >>> np.random.seed(42)
    >>> n, p = 3, 4
    >>> X = np.random.rand(n, p)
    >>> pca = estimators.PCA(1)
    >>> _ = pca.fit(X)
    >>> U, S, V = np.linalg.svd(X)
    >>> pca.P = -pca.P if pca.P[0] < 0.0 else pca.P
    >>> V[[0], :] = -V[[0], :] if V[0, 0] < 0.0 else V[[0], :]
    >>> assert(np.linalg.norm(pca.P - V[[0], :].T) < 5e-8)
    >>>
    >>> np.random.seed(42)
    >>> n, p = 4, 5
    >>> X = np.random.rand(n, p)
    >>> pca = estimators.PCA(2)
    >>> _ = pca.fit(X)
    >>> U, S, V = np.linalg.svd(X)
    >>> pca.P[:, [0]] = -pca.P[:, [0]] if pca.P[0, 0] < 0.0 else pca.P[:, [0]]
    >>> pca.P[:, [1]] = -pca.P[:, [1]] if pca.P[0, 1] < 0.0 else pca.P[:, [1]]
    >>> V[[0], :] = -V[[0], :] if V[0, 0] < 0.0 else V[[0], :]
    >>> V[[1], :] = -V[[1], :] if V[1, 0] < 0.0 else V[[1], :]
    >>> assert(np.linalg.norm(pca.P - V[[0, 1], :].T) < 5e-8)
    >>>
    >>> np.random.seed(42)
    >>> n, p = 5, 6
    >>> X = np.random.rand(n, p)
    >>> pca = estimators.PCA(3, eps=5e-16)  # Note the increased precision!
    >>> _ = pca.fit(X)
    >>> U, S, V = np.linalg.svd(X)
    >>> pca.P[:, [0]] = -pca.P[:, [0]] if pca.P[0, 0] < 0.0 else pca.P[:, [0]]
    >>> pca.P[:, [1]] = -pca.P[:, [1]] if pca.P[0, 1] < 0.0 else pca.P[:, [1]]
    >>> pca.P[:, [2]] = -pca.P[:, [2]] if pca.P[0, 2] < 0.0 else pca.P[:, [2]]
    >>> V[[0], :] = -V[[0], :] if V[0, 0] < 0.0 else V[[0], :]
    >>> V[[1], :] = -V[[1], :] if V[1, 0] < 0.0 else V[[1], :]
    >>> V[[2], :] = -V[[2], :] if V[2, 0] < 0.0 else V[[2], :]
    >>> assert(np.linalg.norm(pca.P - V[[0, 1, 2], :].T) < 5e-14)
    """
    def __init__(self, K, verbose=2,
                 eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):

        super(PCA, self).__init__(verbose=verbose, eps=eps, max_iter=max_iter)

        self.K = max(1, int(K))

        self.reset()

    def reset(self):
        """Resets the estimator such that it is as if just created.

        From BaseEstimator.
        """
        pass

    def fit(self, X, random_state=None):
        """Compute the PCA decomposition of a matrix X.

        X : numpy array
            The matrix to decompose.

        random_state : numpy.random.RandomState
            A random state to use when generating start vectors. Currently not
            used.
        """
        # TODO: Use np.linalg.svd or scipy.sparse.linalg.svds when possible!

        if isinstance(X, list):
            X = X[0]

        P = []
        T = []
        for k in range(self.K):

            p, t = self._compute_component(X)

            P.append(p)
            T.append(t)

            if k < self.K:
                X = self._deflate(X, p, t)

        self.P = np.hstack(P)
        self.T = np.hstack(T)

        return self

    def predict(self, X, return_scores=False):
        """Perform prediction using the fitted parameters.

        Parameters
        ----------
        X : numpy array
            A matrix of new samples to predict in the current PCA model.

        return_scores : bool
            Whether or not to also return the new score matrix. Default is
            false, do not return the score matrix.

        Returns
        -------
        Xhat : numpy array
            The predicted samples using the fitted model.

        That : numpy array
            A column matrix with the computed scores. This matrix is only
            returned if return_scores is True.
        """
        That = np.dot(self.P)
        Xhat = np.dot(That, self.P.T)

        if bool(return_scores):
            return Xhat, That
        else:
            return Xhat

    def score(self, X):
        """Returns a quality measure of the estimators fitted model.

        What is returned depends on the estimator. See the estimator's
        documentation.

        Parameters
        ----------
        X : numpy.ndarray or list of numpy.ndarray
            A single numpy.ndarray, or a list of numpy.ndarray, whose
            prediction using the built model will be scored.

        Returns
        -------
        scores : float or list of float
            The score value (R2), or a list of score values if the input was a
            list.
        """
        nolist = False
        if not isinstance(X, list):
            X = [X]
            nolist = True

        scores = []
        for i in X:
            Xi = X[i]
            Xhati = self.predict(Xi)

            scores.append(1.0 - np.sum((Xi - Xhati)**2.0) / np.sum(Xi**2.0))

        if nolist:
            return scores[0]
        else:
            return scores

    def parameters(self):
        """Returns a dictionary with the estimator's fitted parameters, e.g.
        regression coefficients.

        What is returned depends on the estimator. See the estimator's
        documentation.

        Returns
        -------
        params : dict
            A dictionary with the output parameters of the estimator. For PCA
            this is P, a column matrix of loadings, and T, a column matrix of
            scores. The decomposed matrix is reconstructed by np.dot(T, P.T).
        """
        return {"P": self.P,
                "T": self.T}

    def _start_vector(self, X):

        ssvar = np.sum(X**2.0, axis=0)
        i = np.argmax(ssvar)
        t = X[:, [i]]
        p = np.dot(X.T, t)

        pnorm = np.linalg.norm(p)
        if pnorm > consts.TOLERANCE:
            p = p / pnorm

        return p

    def _compute_component(self, X):
        p = self._start_vector(X)

        for i in range(self.max_iter):
            t = np.dot(X, p)
            p_old = p
            p = np.dot(X.T, t)

            pnorm = np.linalg.norm(p)
            if pnorm > consts.TOLERANCE:
                p = p / pnorm

            if np.linalg.norm(p - p_old) < self.eps:
                break

        return p, t

    def _deflate(self, X, p, t):

        return X - np.dot(t, p.T)


class nPLS(BaseMultiblock, BaseEstimator):
    """The nPLS method for multiblock data analysis.

    Parameters
    ----------
    predComp : list of lists of ints
        The number of joint components between all pairs of blocks.

    precomputedA : ndarray
        A precomputed block covariance matrix.

    numReps : int, optional
        Number of times to recompute the model. The model may have many local
        minima. To avoid using results from suboptimal minima, the best one
        from numReps repeted models will be used. Default is 1.

    randomState : np.random.RandomState, optional
        A random state to use when generating the start vectors. If randomState
        is None, the start vectors will be uniform. Default is None.

    verbose : int, optional
        The verbosity level. Level 0 means print nothing. Level 1 means print
        errors. Level 2 means print errors and warnings. Level 3 means print
        errors, warnings and messages. Default is 2, print errors and warnings.

    eps : float, optional
        Must be positive. The precision used to stop the algorithm. Default is
        consts.TOLERANCE.

    max_iter : int, optional
        Must be positive. The maximum number of iterations to use in the
        algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> import estimators
    >>> np.random.seed(42)
    """
    def __init__(self, predComp, precomputedA=None, numReps=1,
                 randomState=None, verbose=2,
                 eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):

        super(nPLS, self).__init__(verbose=verbose,
                                   eps=eps, max_iter=max_iter)

        self.predComp = predComp
        self.precomputedA = precomputedA
        self.numReps = max(1, int(numReps))
        self.randomState = randomState

        self.reset()

    def reset(self):
        """Resets the estimator such that it is as if just created.

        From BaseEstimator.
        """
        pass

    def f(self, X, W):
        value = np.trace(np.dot(W.T, np.dot(X, W)))

        return value

    def fit(self, X):
        """Fit the estimator to the data.

        Parameters
        ----------
        X : list of ndarrays (n-by-pi)
            The blocks of data. The numpy arrays in X are assumed to have been
            preprocessed appropriately.
        """
        n = len(X)

        # Initialise start vectors
        Z = None
        for i in range(n):
            if self.randomState is None:
                temp = np.ones((1, X[i].shape[1]))
            else:
                temp = self.randomState.rand(1, X[i].shape[1])
            temp = temp / np.linalg.norm(temp)

            if Z is None:
                Z = temp
            else:
                Z = np.hstack((Z, temp))

        # Build block covariance matrix, if not passed to the function
        if self.precomputedA is None:
            A = self.generateA(X, True)  # Make symmetric and positive-definite
        else:
            A = self.precomputedA

        # Find model
        func_val = []
        for it in range(self.max_iter):
            T = Z
            Z = None
            index = 0
            for i in range(n):
                temp = np.dot(A[index:index + X[i].shape[1], :], T.T)
                temp = utils.normaliseColumns(temp)
                if Z is None:
                    Z = temp.T
                else:
                    Z = np.hstack((Z, temp.T))
                index = index + X[i].shape[1]

            func_val.append(self.f(A, Z.T))

            if it >= 1:
                err = func_val[it] - func_val[it - 1]
            else:
                err = func_val[it]

            self.num_iter = it + 1

            if abs(err) < consts.TOLERANCE:
                break

        # Find all model vectors
        index = 0
        W = [0] * n
        T = [0] * n
        P = [0] * n
        for i in range(n):
            W[i] = Z[:, index:index + X[i].shape[1]].T
            T[i] = np.dot(X[i], W[i])
            P[i] = np.dot(X[i].T, T[i])
            toto = np.dot(T[i].T, T[i])
            if toto > consts.TOLERANCE:
                P[i] = P[i] / toto
            else:
                self.warn("Too small joint component for matrix %d! "
                          "Trying to continue!" % (i,))

            # Normalise P? It matters whether we update with W or with P!
            # TODO: Option?
#            normp = norm(P{i});
#            P{i} = P{i} / normp;
#            T{i} = T{i} * normp;

            index = index + X[i].shape[1]

        self.func_val = func_val
        self.W = W
        self.T = T
        self.P = P

        return self

    def predict(self, X, which=[], return_scores=False):
        """Perform prediction using the fitted parameters.

        From BaseEstimator.

        Parameters
        ----------
        X : list of numpy.ndarrays
            A containing the blocks of data.

        which : list of int, optional
            The indices of the blocks to predict. Default is [], an empty list,
            which means that all blocks are predicted by the connected blocks
            (defined in self.predComp).

        return_scores : bool
            Whether or not to also return the predicted score matrices. Default
            is false, do not return the score matrices.

        Returns
        -------
        Xhat : list of numpy.ndarray
            The predicted samples using the fitted models. The returned list
            has length len(which), and "which" indexes the matrices that was
            predicted.

        That : numpy.ndarray
            A list of column matrices with the computed scores. This matrix is
            only returned if return_scores is True.
        """
        return_scores = bool(return_scores)
        n = len(X)

        T = [None] * n
        for i in range(n):
            Xi = X[i]
            Wi = self.W[i]
            Pi = self.P[i]
            ki = Wi.shape[1]
            Ti = []
            for k in range(ki):
                t = np.dot(Xi, Wi[:, [k]])
                Xi = Xi - np.dot(t, Pi[:, [k]].T)
                Ti.append(t)
            T[i] = np.hstack(Ti)

        if len(which) == 0:
            which = list(range(n))

        numComp = T[0].shape[1]
        Xhat = [0.0] * len(which)
        That = [0.0] * len(which)
        for w in range(len(which)):
            j = which[w]
            Tw = T[j]
            Pw = self.P[j]

            Xhatw = np.zeros(X[j].shape)
            Thatw = []
            for k in range(numComp):
                Tiks = []
                for i in range(n):
                    if self.predComp[i][j] > 0:  # If i predicts j
                        Ti = T[i]
                        Tik = Ti[:, [k]]
                        Tiks.append(Tik)
                if len(Tiks) > 0:
                    Tiks = np.hstack(Tiks)
                    beta = np.dot(np.linalg.pinv(Tiks), Tw[:, [k]])
                    Thatwk = np.dot(Tiks, beta)
                    Xhatw = Xhatw + np.dot(Thatwk, Pw.T)

                    if return_scores:
                        Thatw.append(Thatwk)

            Xhat[w] = Xhatw
            if return_scores:
                That[w] = np.hstack(Thatw)

        if return_scores:
            return Xhat, That
        else:
            return Xhat

    def score(self, X):
        """Computes the sum of predicted R², i.e. the sum of how well each
        block is predicted by their connected blocks.

        From BaseEstimator.

        Parameters
        ----------
        X : list of numpy.ndarray
            A list of numpy.ndarray. The matrices that will be predicted and
            scored.

        Returns
        -------
        scores : float
            The sum of the predicted R² for all blocks.
        """
        Xhat = self.predict(X)

        n = len(X)
        scores = []
        for i in range(n):
            Xi = X[i]
            Xhati = Xhat[i]

            scores.append(1.0 - np.sum((Xi - Xhati)**2.0) / np.sum(Xi**2.0))

        return np.sum(scores) / float(n)

    def parameters(self):
        """Returns a dictionary with the estimator's fitted parameters, e.g.
        regression coefficients.

        What is returned depends on the estimator. See the estimator's
        documentation.

        From BaseEstimator.
        """
        return {"W": self.W,
                "T": self.T,
                "P": self.P}

    def generateA(self, X, psd=True):
        """Generates the block covariance matrix A.

        Parameters
        ----------
        X : ndarray
            The block covariance matrices consists of covariance matrices
            computed from pairs of matrices, from the list of matrices in X.

        psd : bool
            Whether or not to force the generated matrix to be positive
            semi-definite. Default is True, force it to be positive
            semi-definite.
        """
        n = len(X)
        A = None
        for i in range(n):
            R = None
            for j in range(n):
                if self.predComp[i][j] == 0:
                    if R is None:
                        R = np.zeros((X[i].shape[1], X[j].shape[1]))
                    else:
                        R = np.hstack((R, np.zeros((X[i].shape[1],
                                                    X[j].shape[1]))))
                else:
                    if R is None:
                        R = np.dot(X[i].T, X[j])
                    else:
                        R = np.hstack((R, np.dot(X[i].T, X[j])))
            if A is None:
                A = R
            else:
                A = np.vstack((A, R))

        if psd:
            A = (A + A.T) / 2.0  # Make it symmetric
            [D, _] = np.linalg.eig(A)
            sigma = np.min(np.real(D))
            if sigma < 0.0:  # It is negative definite
                # Make it positive semi-definite:
                A = A + np.eye(*A.shape) * sigma * -n

        return A


class OnPLS(BaseMultiblock, BaseEstimator):
    """The OnPLS method for multiblock data analysis.

    Parameters
    ----------
    predComp : list of lists of ints
        The number of joint components between all pairs of blocks.

    orthComp : list of int
        The number of non-joint components for each block.

    model : OnPLS.ModelType
        The kind of OnPLS model to build. The alternatives are:

            1. OnPLS.ModelType.LP_NONE,
            2. OnPLS.ModelType.LP_FULL,
            3. OnPLS.ModelType.LP_PARTIAL.

        Default is OnPLS.ModelType.LP_NONE.

    precomputedW : list of ndarray, optional
        Supply the globally joint components in order to avoid recomputing them
        on each run. Default is None, meaning that no precomputed matrices are
        available.

    numReps : int, optional
        Number of times to recompute the model. The model has many local
        minima. To avoid using results from suboptimal minima, the best one
        from numReps repeted models will be used. Default is 1.

    verbose : int, optional
        The verbosity level. Level 0 means print nothing. Level 1 means print
        errors. Level 2 means print errors and warnings. Level 3 means print
        errors, warnings and messages. Default is 2, print errors and warnings.

    eps : float, optional
        Must be positive. The precision used to stop the algorithm. Default is
        consts.TOLERANCE.

    max_iter : int, optional
        Must be positive. The maximum number of iterations to use in the
        algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> import OnPLS.estimators as estimators
    >>> np.random.seed(42)
    >>> n, p_1, p_2, p_3 = 10, 5, 10, 15
    >>> t = np.sort(np.random.randn(n, 1), axis=0)
    >>> p1 = np.sort(np.random.randn(p_1, 1), axis=0)
    >>> p2 = np.sort(np.random.randn(p_2, 1), axis=0)
    >>> p3 = np.sort(np.random.randn(p_3, 1), axis=0)
    >>> X1 = np.dot(t, p1.T) + 0.1 * np.random.randn(n, p_1)
    >>> X2 = np.dot(t, p2.T) + 0.1 * np.random.randn(n, p_2)
    >>> X3 = np.dot(t, p3.T) + 0.1 * np.random.randn(n, p_3)
    >>> predComp = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    >>> orthComp = [1, 1, 1]
    >>> model = None
    >>> precomputedW = None
    >>> onpls = estimators.OnPLS(predComp, orthComp, model, precomputedW,
    ...     numReps=1, verbose=1)
    >>> _ = onpls.fit([X1, X2, X3])
    """
    def __init__(self, predComp, orthComp, model=None,
                 precomputedW=None, numReps=1, verbose=2,
                 eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):

        super(OnPLS, self).__init__(verbose=verbose,
                                    eps=eps, max_iter=max_iter)

        self.predComp = predComp
        self.orthComp = orthComp
        self.model = model
        self.precomputedW = precomputedW
        self.numReps = max(1, int(numReps))

        self.reset()

    def reset(self):
        """Resets the estimator such that it is as if just created.

        From BaseEstimator.
        """
        if hasattr(self, "W"):
            del self.W
        if hasattr(self, "T"):
            del self.T
        if hasattr(self, "P"):
            del self.P

    def fit(self, X):
        """Fit the estimator to the data.

        Parameters
        ----------
        X : list of ndarrays (n-by-pi)
            The blocks of data. The numpy arrays in X are assumed to have been
            preprocessed appropriately.
        """
        if self.precomputedW is None:
            self.output("Creating combined loadings ...")
            self.precomputedW = self.computeGloballyJointW(X)
            self.output("Combined loadings created!")

        ok = True
        okPred = True

        self.output("Filtering matrices ...")
        n = len(X)
        Wo = [None] * n
        To = [None] * n
        Po = [None] * n
        ssX = [0] * n
        for i in range(n):
            Xi = X[i]
            gjWi = self.precomputedW[i]
            ssX[i] = utils.sumOfSquares(Xi)

            # If we have no vectors here, we have no predictive model for
            # this matrix, and thus no overlapping variation either:
            if gjWi.shape[1] < 1:
                continue

            for c in range(self.orthComp[i]):
                Ti = np.dot(Xi, gjWi)  # Recall that Wi'*Wi = I
                # Any row vector in E represents a possible wortho vector
                Ei = Xi - np.dot(Ti, gjWi.T)
                Wortho = np.dot(Ti.T, Ei)

                if utils.sumOfSquares(Wortho) / ssX[i] < consts.TOLERANCE:
                    self.warn("No more orthogonal components appear to "
                              "exist for matrix %d! Trying anyway!" % (i,))

                pca = PCA(1, eps=self.eps, max_iter=self.max_iter)
                pca.fit(Wortho)
                wortho = pca.P
                # We can also use Ei here. Doesn't matter.
                tortho = np.dot(Xi, wortho)
                portho = np.dot(Xi.T, tortho)
                toto = np.dot(tortho.T, tortho)[0, 0]
                # Avoid division by zero by defining 0 / 0 = 0
                if toto > consts.TOLERANCE:
                    portho = portho / toto
                else:
                    self.warn("Too small orthogonal component for matrix %d! "
                              "Not included!" % (i,))
                    break

                # It has been discussed whether or not to normalise portho
#                 normpo = np.linalg.norm(portho)
#                 tortho = tortho * normpo
#                 portho = portho / normpo

                if utils.sumOfSquares(
                        np.dot(tortho, portho.T)) / ssX[i] < consts.LIMIT_R2:
                    self.warn("Too small orthogonal component for matrix %d! "
                              "Not included!" % (i,))
                    ok = False
                    break

                if self.verbose == 3:
                    self.output("Matrix %d, orthogonal component %d, "
                                "T'*to overlap: %f, norm po: %f, "
                                "R2(to*po'): %f"
                                % (i, c, np.dot(tortho.T,
                                                np.dot(np.dot(Ti, Ti.T),
                                                       tortho)),
                                   np.linalg.norm(portho),
                                   utils.sumOfSquares(
                                       np.dot(tortho, portho.T) / ssX[i])))

                # Filter matrix by removing orthogonal variation:
                Xi = Xi - np.dot(tortho, portho.T)

                # Save results
                if Wo[i] is None:
                    Wo[i] = wortho
                else:
                    Wo[i] = np.hstack((Wo[i], wortho))
                if To[i] is None:
                    To[i] = tortho
                else:
                    To[i] = np.hstack((To[i], tortho))
                if Po[i] is None:
                    Po[i] = portho
                else:
                    Po[i] = np.hstack((Po[i], portho))

            X[i] = Xi
        self.output("Matrices filtered!")

        self.output("Building globally joint OnPLS model ...")
        W = [None] * n
        T = [None] * n
        P = [None] * n
        ftot = 0.0
        numGlobalPredComp = utils.leastNonZero(self.predComp)
        for comp in range(numGlobalPredComp):
            self.output("Calculating component %d ..." % (comp + 1,))

            fMax = -np.inf
            for i in range(self.numReps):
                # Find filtered nPLS component:
                # TODO: Precompute A!
                npls = nPLS(self.predComp)
                npls.fit(X)
                _Wx = npls.W
                _Tx = npls.T
                _Px = npls.P
                its = npls.num_iter
                func_val = npls.func_val
                # TODO: Use nPLS.f(...) instead:
                s = np.sum(utils.sumOfCovariances(_Tx, self.predComp))
                if i == 0 or s > fMax:
                    fMax = s
                    Wx = _Wx
                    Tx = _Tx
                    Px = _Px

            if len(func_val) > 1:
                maxerr = func_val[-1] - func_val[-2]
            else:
                maxerr = 0.0

            f = 0.0
            for i in range(n):
                if its >= consts.MAX_ITER or maxerr >= consts.TOLERANCE:
                    self.warn("Component %d may not be correct or is "
                              "non-significant, the number of iterations "
                              "was: %d (maxiter=%d), error is: %.3g "
                              "(noiselevel=%.3g)."
                              % (comp, its, consts.MAX_ITER, maxerr,
                                 consts.TOLERANCE))
                    okPred = False

                # Can also update with W but that gives non-orthogonal scores!
                # TODO: Option?
                temp = np.dot(Tx[i], Px[i].T)
                ssTemp = utils.sumOfSquares(temp)
                if ssTemp / ssX[i] < consts.LIMIT_R2:
                    okPred = False
                    for j in range(i - 1):
                        # Remove already saved results
                        try:
                            np.delete(W[j], comp, axis=1)
                        except:
                            pass
                        try:
                            np.delete(T[j], comp, axis=1)
                        except:
                            pass
                        try:
                            np.delete(P[j], comp, axis=1)
                        except:
                            pass
                    if comp >= 1:
                        self.warn("Component is not significant! Contribution "
                                  "of component %d in matrix %d is "
                                  "%.2f %% < 1 %%. Not included!"
                                  % (comp, i, 100 * ssTemp / ssX[i]))
                        break
                    else:
                        self.warn("Component is not significant! Contribution "
                                  "of component %d in matrix %d is "
                                  "%.2f %% < 1 %%. Warning, was included!"
                                  % (comp, i, 100 * ssTemp / ssX[i]))

                # Save results
                if W[i] is None:
                    W[i] = Wx[i]
                else:
                    W[i] = np.hstack((W[i], Wx[i]))
                if T[i] is None:
                    T[i] = Tx[i]
                else:
                    T[i] = np.hstack((T[i], Tx[i]))
                if P[i] is None:
                    P[i] = Px[i]
                else:
                    P[i] = np.hstack((P[i], Px[i]))

                # Deflate joint variation from matrix
                X[i] = X[i] - temp

                # TODO: Create method for objective function
                for j in range(n):
                    if self.predComp[i][j] > 0:
                        f = f + np.dot(Tx[i].T, Tx[j])

            if comp > 1 and (not okPred):
                break

            ftot = ftot + f
            self.output("DONE! f = %.8f" % (f,))

        self.Wo = Wo
        self.To = To
        self.Po = Po
        self.W = W
        self.T = T
        self.P = P

        numGlobalPredComp = self.T[0].shape[1]
        self.output("OnPLS model created with %d components! ftot = %f"
                    % (numGlobalPredComp, ftot))

        return self

    def predict(self, X, which=[], return_scores=False):
        """Perform prediction using the fitted parameters.

        From BaseEstimator.

        Parameters
        ----------
        X : list of numpy.ndarrays
            A containing the blocks of data.

        which : list of int, optional
            The indices of the blocks to predict. Default is [], an empty list,
            which means that all blocks are predicted by the connected blocks
            (defined in self.predComp).

        return_scores : bool
            Whether or not to also return the predicted score matrices. Default
            is false, do not return the score matrices.

        Returns
        -------
        Xhat : list of numpy.ndarray
            The predicted samples using the fitted models. The returned list
            has length len(which), and "which" indexes the matrices that was
            predicted.

        That : numpy.ndarray
            A list of column matrices with the computed scores. This matrix is
            only returned if return_scores is True.
        """
        return_scores = bool(return_scores)
        n = len(X)

        # Filter test data
        for i in range(n):
            Xi = X[i]
            woi = self.Wo[i]
            if woi is None:
                continue
            poi = self.Po[i]
            ki = self.Wo[i].shape[1]
            for k in range(ki):
                woik = woi[:, [k]]
                poik = poi[:, [k]]
                to = np.dot(Xi, woik)
                Xi = Xi - np.dot(to, poik.T)

            X[i] = Xi

        T = [None] * n
        for i in range(n):
            Xi = X[i]
            Wi = self.W[i]
            Pi = self.P[i]
            ki = Wi.shape[1]
            Ti = []
            for k in range(ki):
                t = np.dot(Xi, Wi[:, [k]])
                Xi = Xi - np.dot(t, Pi[:, [k]].T)
                Ti.append(t)
            T[i] = np.hstack(Ti)

        if len(which) == 0:
            which = list(range(n))

        numComp = T[0].shape[1]
        Xhat = [0.0] * len(which)
        That = [0.0] * len(which)
        for j in range(len(which)):
            w = which[j]
            Tw = T[w]
            Pw = self.P[w]

            Xhatw = np.zeros(X[w].shape)
            Thatw = []
            for k in range(numComp):
                Tiks = []
                for i in range(n):
                    if self.predComp[i][w] > 0:  # If i predicts w
                        Ti = T[i]
                        Tik = Ti[:, [k]]
                        Tiks.append(Tik)
                Tiks = np.hstack(Tiks)
                if len(Tiks) > 0:
                    beta = np.dot(np.linalg.pinv(Tiks), Tw[:, [k]])
                    Thatwk = np.dot(Tiks, beta)
                    Xhatw = Xhatw + np.dot(Thatwk, Pw.T)

                    if return_scores:
                        Thatw.append(Thatwk)

            Xhat[j] = Xhatw
            if return_scores:
                That[j] = np.hstack(Thatw)

        if return_scores:
            return Xhat, That
        else:
            return Xhat

    def score(self, X):
        """Computes the mean predicted R² for all blocks, i.e. the mean of how
        well each block is predicted by their connected blocks.

        From BaseEstimator.

        Parameters
        ----------
        X : list of numpy.ndarray
            A list of numpy.ndarray. The matrices that will be predicted and
            scored.

        Returns
        -------
        scores : float
            The mean of the predicted R² for all blocks.
        """
        Xhat = self.predict(X)

        n = len(X)
        scores = []
        for i in range(n):
            Xi = X[i]
            Xhati = Xhat[i]

            scores.append(1.0 - np.sum((Xi - Xhati)**2.0) / np.sum(Xi**2.0))

        return np.sum(scores) / float(n)

    def parameters(self):
        """Returns a dictionary with the estimator's fitted parameters, e.g.
        regression coefficients.

        What is returned depends on the estimator. See the estimator's
        documentation.

        From BaseEstimator.
        """
        return {"W": self.W,
                "T": self.T,
                "P": self.P,
                "Wo": self.Wo,
                "To": self.To,
                "Po": self.Po}

    def computeGloballyJointW(self, X, returnComputedPredComp=False):

        n = len(X)
        computedPredComp = np.zeros((n, n)).tolist()
        gjW = [0] * n
        for i in range(n):
            W = None
            for j in range(n):
                if self.predComp[i][j] > 0:
                    pca = PCA(self.predComp[i][j],
                              eps=self.eps, max_iter=self.max_iter)
                    # TODO: We don't actually need to compute the inner product
                    XjtXi = np.dot(X[j].T, X[i])
                    pca.fit(XjtXi)
                    if W is None:
                        W = pca.P
                    else:
                        W = np.hstack((W, pca.P))

                    computedPredComp[i][j] = W.shape[1]
                else:
                    computedPredComp[i][j] = 0

            U, S, V = np.linalg.svd(W)
            rank_thresh = np.max(S) * np.max(S.shape) * np.finfo(W.dtype).eps
            rank_W = np.sum(S > rank_thresh)

            comps = utils.leastNonZero(self.predComp[i])
            comps = min(rank_W, max(1, comps))

            W = W[:, 0:comps]
            if W.shape[1] > 0:
                W = utils.normaliseColumns(W)

            gjW[i] = W

        if returnComputedPredComp:
            return gjW, computedPredComp
        else:
            return gjW
