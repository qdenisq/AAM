from __future__ import print_function
from __future__ import division
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
            print("\rtraining progress: {} out of {}".format(i+1, n_iterations), end='')
            target = input[:, np.random.randint(0, high=input.shape[1])] if shuffle else input[:, i % input.shape[1]]
            bmu, bmu_idx = self.find_bmu(target)
            assert bmu_idx is not None, "bmu is none for the input: {}".format(target)
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


class MultilayerSequenceSom:
    def __init__(self, layers_shapes, sequences_lengths, input_vector_length):
        assert len(layers_shapes) == len(sequences_lengths), \
            "len of layers %d and len of sequences %d don't match" % (len(layers_shapes), len(sequences_lengths))
        self.shapes = layers_shapes
        self.sequences_lengths = sequences_lengths
        self.input_length = input_vector_length
        self.soms = []
        for i, shape in enumerate(self.shapes):
            # create first som layer manually
            if i == 0:
                som = Som(shape, self.input_length * sequences_lengths[0])
                self.soms.append(som)
            else:
                # input vector for all soms except first one is sequence of units coordinates form previous som
                # i.e. for 2d som with 3 units in sequence input_vector = [x0, y0, x1, y1, x2, y2]
                som = Som(shape, len(self.shapes[i-1]) * sequences_lengths[i])
                self.soms.append(som)

    def train(self, data, n_iterations, learning_rates, dump=True):
        """
        Train network with data. Each layer is trained for n_iterations. Training is done consequently for all layers in
         down to top order
        :param data: training data
        :param n_iterations:list of number of training iterations for each layer
        :return:
        """

        n_soms = len(self.soms)
        input_data = data
        for i, som in enumerate(self.soms):
            # training
            print("     Training {} som out of {}".format(i + 1, n_soms))
            self.soms[i].train(input_data, n_iterations[i], learning_rates[i], shuffle=False)

            if i == (n_soms - 1):
                print("\nTraining successfully finished")
                return

            # prepare training data for the next layer

            print("\n     Prepare training data for {} layer out of {}".format(i + 2, n_soms))
            output = []
            for j in range(input_data.shape[1]):
                print('\r{} out of {}'.format(j+1, input_data.shape[1]), end='')
                input_sample = input_data[:, j]
                bmu, bmu_idx = self.soms[i].find_bmu(input_sample)
                output.append([bmu_idx[idx] / self.shapes[i][idx] for idx in range(len(bmu_idx))])

            output = np.array(output)
            if dump:
                print("\nSaving output...")
                utils.save_obj(output, "mssom_l{}_training_output".format(i))
            sequence_length = self.sequences_lengths[i+1]
            n = len(output) // self.sequences_lengths[i+1]
            output = [np.array(v).reshape((sequence_length * len(self.shapes[i]))) for v in
                                  np.array_split(output[:sequence_length * n], n)]
            input_data = np.array(output).transpose()

            if dump:
                print("Saving training_input...")
                utils.save_obj(input_data, "mssom_l{}_training_input".format(i+1))

        if dump:
            print("\nSaving network...")
            utils.save_obj(self, "mssom")

        print("\n\nTraining successfully finished")
        return

    def find_bmu(self, data):
        """
        return list of best matching units of all layers in the mssom
        :param data: input has to be 2d np.array with the shape of input_length in the 1st dim and len of maximum sequence
        in the 2nd dim
        :return: bmus: list of arrays of bmus of each layer in the mssom
        """
        bmus = []
        input_data = data
        for i, som in enumerate(self.soms):
            output = []
            for j in range(input_data.shape[1]):
                _, bmu_idx = som.find_bmu(input_data[:, j])
                output.append([bmu_idx[idx] / self.shapes[i][idx] for idx in range(len(bmu_idx))])
            output = np.array(output)
            bmus.append(output)
            sequence_length = self.sequences_lengths[i + 1]
            n = len(output) // self.sequences_lengths[i + 1]
            output = [np.array(v).reshape((sequence_length * len(self.shapes[i]))) for v in
                      np.array_split(output[:sequence_length * n], n)]
            input_data = np.array(output).transpose()
        return bmus