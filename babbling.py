from numpy.random import normal, uniform
import numpy as np
import utils

from scipy.io import wavfile

class Babbler:
    def __init__(self, num_params=30, fixed_params_idx=list(range(24, 30)),
                 fixed_params_values=[0.14285714285714285,
                                      0.5,
                                      0.17142857142857143,
                                      0.17142857142857143,
                                      0.16666666666666669,
                                      0.0]):

        self.fixed_params_values = fixed_params_values
        self.gmm = []
        self.gmm_weights = []

        self.gmm.append(lambda: uniform(size=num_params))
        self.gmm_weights.append(1.0)

        self.num_params = num_params
        self.fixed_params_idx = fixed_params_idx

        self.test_data = []
        return

    def learn(self, test_data):
        self.test_data = test_data

        for i in range(50):
            print "{0}:",i
            cf = self.explore()
            dist = self.evaluate(cf)
            if dist < 4.0:
                print "converged: ", np.around(cf, decimals=4)
                utils.save_obj(cf, "cf_a")
                audio = utils.synthesize_static_sound(cf)
                wav = np.array(audio)
                wav_int = np.int16(wav * (2 ** 15 - 1))
                wavfile.write('Obj/a.wav', 22050, wav_int)
                return
        # repeat explore and evaluate
        return

    def explore(self):
        cumsum = np.cumsum(self.gmm_weights)
        rnd = uniform()
        idx = np.searchsorted(cumsum, rnd)

        # explore from gmm[idx]
        cf = self.sample(idx)
        # print idx, rnd
        return cf

    def sample(self, idx):
        cf = self.gmm[idx]()
        return [cf[i] if i not in self.fixed_params_idx else self.fixed_params_values[self.fixed_params_idx.index(i)]
                for i in range(self.num_params)]

    def evaluate(self, cf):
        if not self.test_data:
            raise "no test data"
        # sinthesize sound based on cf

        audio = np.array(utils.synthesize_static_sound(cf))
        audio_int = np.int16(audio * (2 ** 15 - 1))
        mfcc = np.array(utils.calc_mfcc_from_vowel(audio_int))

        # calc average distance to all units in test_data
        distance = [np.linalg.norm(mfcc - np.array(t)) for t in self.test_data]
        average_distance = np.mean(distance)
        # calculate goodness of exloration
        goodness = np.exp(-1.0 * average_distance / 4.0)
        # assign weight for new gmm component
        w = goodness
        center = cf
        covariance = np.identity(self.num_params) * (0.1 * (1.0 - goodness))
        new_comp = lambda: np.random.multivariate_normal(center, covariance)

        self.gmm_weights.append(w)
        self.gmm_weights = [w/sum(self.gmm_weights) for w in self.gmm_weights]
        self.gmm.append(new_comp)
        print "distance:", average_distance
        print "weights: ", np.around(self.gmm_weights, decimals=2)

        return average_distance

