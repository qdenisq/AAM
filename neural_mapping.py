from numpy.random import normal, uniform
from scipy.stats import multivariate_normal
import numpy as np
import utils
import copy


class NeuralMap:
    def __init__(self, dim):
        self.dim = dim
        self.weights = np.array([])
        self.means = np.array([])
        self.covs = np.array([])
        return

    def add(self, w, mean, cov):
        """Add Gaussian component to the GM
           :param w: weighting coefficient
           :param mean: center (mean) of the component
           :param cov: covariance matrix
        """
        assert len(mean) == len(cov) == self.dim
        self.weights = self.weights.append(w)
        self.means = self.means.append(mean, axis=0)
        self.covs = self.covs.array(cov, axis=0)
        return

    def propagate_gm_signal(self, w, m, cov, idx=[]):
        """Propagates the signal through the neural map, signal has a form Gaussian Mixture.
        Note that output is also GM.
           :param w: array of weighting coefficients
           :param m: array of centers (mean) of the gm components
           :param cov: covariance matrix array
           :param idx: array of dimension indexes the signal comes from
           :return weights: array of weighting coefficients
           :return means: array of centers (mean) of the gm components
           :return covs: covariance matrix array
        """
        w_3arr = []
        mu_3arr = []
        cov_3arr = []
        for w_1, mu_1, cov_1 in zip(self.weights, self.means, self.covs):
            for w_2, mu_2, cov_2 in zip(w, m, cov):
                w_3, mu_3, cov_3 = NeuralMap.calc_response(w_1, mu_1, cov_1, w_2, mu_2, cov_2, idx)
                w_3arr.append(w_3)
                mu_3arr.append(mu_3)
                cov_3arr.append(cov_3)
        return w_3arr, mu_3arr, cov_3arr

    @staticmethod
    def calc_response(w_1, mu_1, cov_1, w_2, mu_2, cov_2, idx=[]):
        """Calculate the propagation of Gaussian signal through another Gaussian. Output is also Gaussian
        :param w_1:
        :param mu_1:
        :param cov_1:
        :param w_2:
        :param mu_2:
        :param cov_2:
        :param idx: array of dimension indexes the signal comes from
        :return:
        """

        input_idx = idx if idx is not [] else np.array(list(range(len(mu_2))))
        output_idx = np.array([i not in input_idx for i in range(len(mu_1))])

        m = len(input_idx)

        mu_3 = mu_1[output_idx]
        cov_3 = cov_1[output_idx]

        A = cov_1[input_idx]
        a = mu_1[input_idx]

        B = cov_2
        b = mu_2

        AB = np.add(A, B)
        invAB = np.array([1/v for v in AB])
        detAB = np.prod(AB)

        absub = np.subtract(a, b)
        c = sum(absub[i]**2 * invAB[i] for i in range(m))
        z = 1. / np.sqrt(detAB * (2*np.pi)**m) * np.exp(-0.5 * c)
        w_3 = w_1 * w_2 * z
        return w_3, mu_3, cov_3

    
