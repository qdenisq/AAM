from __future__ import print_function
from neural_mapping import NeuralMap
import numpy as np
import utils



class GSom(NeuralMap):

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
                if w_3 > 1.:
                    w_3 = 1.

                delta_mu = delta_mu * w_3 * 0.05
                self.means[i][:self.input_dim] += delta_mu
                # also we need to adjust means of the adjacent components to be selective to this signal as well
                for j in range(len(self.weights)):
                    delta_mu = mu_2 - self.means[j][:self.input_dim]
                    if w_3 > 1.:
                        w_3 = 1.
                    delta_mu = delta_mu * w_3 * 0.05
                    delta_mu *= np.exp(-1. * (np.linalg.norm(self.means[j][self.input_dim:] - mu_3)**2) / neighbour_sigma**2)
                    self.means[j][:self.input_dim] += delta_mu


class Som:
    def __init__(self, network_shape, input_shape):
        self.network_shape = network_shape
        self.input_shape = input_shape
        self.net = np.random.random(np.append(network_shape, input_shape))
        normal = lambda vec: [x/sum(vec) for x in vec]
        # self.net = np.apply_along_axis(normal, len(network_shape), self.net)
        self.radius = max(network_shape) / 2
        return

    def train(self, input, n_iterations, learning_rate, shuffle=True):
        time_constant = n_iterations / np.log(self.radius)
        self.learning_rate = learning_rate
        radius = self.radius
        for i in range(n_iterations):
            # sys.stdout.flush()
            print("\repoch out of {}: {}".format(n_iterations, i+1), end='')
            target = input[:, np.random.randint(0, high=input.shape[1])] if shuffle else input[:, i % input.shape[1]]
            bmu, bmu_idx = self.find_bmu(target)
            self.update_weights(target, bmu_idx, radius, learning_rate)
            learning_rate = self.decay_learning_rate(i, n_iterations)
            radius = self.decay_radius(i, time_constant)

    def find_bmu(self, t):
        min_dist = np.iinfo(np.int).max
        max_resp = 0.
        bmu_idx = None
        for i in np.ndindex(*self.network_shape):
            w = self.net[i].reshape(self.input_shape)
            resp = np.dot(w, t) / np.linalg.norm(t) / np.linalg.norm(w)
            sq_dist = np.sum((w - t) **2)
            if resp > max_resp:
                max_resp = resp
                bmu_idx = i
            # if sq_dist < min_dist:
            #     min_dist = sq_dist
            #     bmu_idx = i
        bmu = self.net[bmu_idx]
        return (bmu, bmu_idx)

    def decay_radius(self, i, time_constant):
        return self.radius * np.exp(-i / time_constant)

    def decay_learning_rate(self, i, n_iterations):
        return self.learning_rate * np.exp(-i / n_iterations)

    def calc_influence(self, dist, radius):
        return np.exp(-dist / (2. * (radius**2)))

    def update_weights(self, target, bmu_idx, radius, learning_rate):

        for i in np.ndindex(*self.network_shape):
            dist = np.sum(np.subtract(bmu_idx, i)**2)
            if dist <= radius**2:
                influence = self.calc_influence(dist, radius)
                w = self.net[i]
                delta_w = learning_rate * influence * (target - w)
                self.net[i] = w + delta_w
                # self.net[i] = [x/sum(self.net[i]) for x in self.net[i]]

