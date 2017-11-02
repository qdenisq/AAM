from neural_mapping import NeuralMap
import numpy as np


class Som(NeuralMap):

    def __init__(self, num_input_dim, map_dim):
        NeuralMap.__init__(self, num_input_dim + map_dim)
        self.input_dim = num_input_dim
        self.map_dim = map_dim
        return

    def train(self, input_w, input_mu, input_cov, neighbour_sigma):
        """Propagates the signal through the neural map and adjust its components to hebbian rule, signal has a form of a Gaussian Mixture.
                Note that output is also GM.
                   :param input_w: array of weighting coefficients
                   :param input_mu: array of centers (mean) of the gm components
                   :param input_cov: covariance matrix array
                   :return weights: array of weighting coefficients
                   :return means: array of centers (mean) of the gm components
                   :return covs: covariance matrix array
                """
        for w_1, mu_1, cov_1, i in zip(self.weights, self.means, self.covs, range(len(self.weights))):
            for w_2, mu_2, cov_2 in zip(input_w, input_mu, input_cov):
                w_3, mu_3, cov_3 = self.calc_response(w_1, mu_1, cov_1, w_2, mu_2, cov_2, range(self.input_dim))
                # adjust mu_1 with regard to the response and mu_2
                delta_mu = mu_2 - mu_1[:self.input_dim]
                if w_3 < 1.:
                    delta_mu = delta_mu * w_3
                self.means[i][:self.input_dim] += delta_mu
                # also we need to adjust means of the adjacent components to be selective to this signal as well
                for j in range(self.weights):
                    delta_mu = mu_2 - self.means[j][:self.input_dim]
                    if w_3 < 1.:
                        delta_mu = delta_mu * w_3
                    delta_mu *= np.exp(-1. * (np.linalg.norm(self.means[j][self.input_dim:] - mu_3)**2) / neighbour_sigma**2)
                    self.means[j][:self.input_dim] += delta_mu