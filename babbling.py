from numpy.random import normal, uniform
from scipy.stats import multivariate_normal
import numpy as np
import utils
import copy

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
        self.gmm_pdf = []

        self.gmm.append(lambda: uniform(size=num_params))
        self.gmm_weights.append(1.0)
        self.gmm_pdf.append(lambda x: self.gmm_weights[0])

        self.num_params = num_params
        self.fixed_params_idx = fixed_params_idx

        self.test_data = []
        self.target = []
        self.norm_coef = 0
        return

    def learn(self, test_data, name, num_iterations=500, dump=False):
        self.test_data = test_data
        self.target = np.mean(np.array(test_data), axis=0)
        self.norm_coef = max(self.target)
        self.target = np.array([v / self.norm_coef for v in self.target])

        min_dist = 100.0
        best_cf = []
        dump_components=[]
        competence = []
        for i in range(num_iterations):
            print "iteration {0}:".format(i)
            cf = self.explore()
            dist, mfcc_norm, weight, center, covariance = self.evaluate(cf)



            if dump is True:
                dump_components.append((weight, center, covariance))
            if dist < min_dist:
                best_cf = cf
                min_dist = dist

                if dist < 0.01:
                    print "converged: ", np.around(cf, decimals=4)
                    utils.save_obj(cf, "cf_{0}".format(name))

                    audio = utils.synthesize_static_sound(cf)
                    wav = np.array(audio)
                    wav_int = np.int16(wav * (2 ** 15 - 1))
                    wavfile.write('Obj/{0}.wav'.format(name), 22050, wav_int)
                    if dump is True:
                        utils.save_obj(dump_components, "components_{0}".format(name))
                        utils.save_obj(competence, "competence_{0}".format(name))
                    return
            competence.append(np.exp(-1. * min_dist))
        # repeat explore and evaluate
        utils.save_obj(best_cf, "best_cf_{0}".format(name))

        audio = utils.synthesize_static_sound(best_cf)
        wav = np.array(audio)
        wav_int = np.int16(wav * (2 ** 15 - 1))
        print "best distance: ", min_dist, np.around(best_cf, decimals=4)

        utils.save_obj(best_cf, "cf_{0}".format(name))

        wavfile.write('Obj/{0}.wav'.format(name), 22050, wav_int)
        if dump is True:
            utils.save_obj(dump_components, "components_{0}".format(name))
            utils.save_obj(competence, "competence_{0}".format(name))
        return

    def explore(self):
        # find all components with positive weights
        pos_idx = [j for j, e in enumerate(self.gmm_weights) if e >= 0]
        neg_idx = [j for j, e in enumerate(self.gmm_weights) if e < 0]
        cumsum = np.cumsum([self.gmm_weights[j] for j in pos_idx])
        accepted = False
        while not accepted:
            # choose a component to generate from
            rv = uniform(low=0.0, high=cumsum[-1], size=1)
            idx = np.searchsorted(cumsum, rv)
            idx = pos_idx[idx]
            # generate rnd number
            cf = self.sample(idx)
            if any(v > 1.0 or v < 0.0 for v in cf):
                continue
            # find probability for given rnd number
            p_pos = sum([self.gmm_pdf[j](cf) for j in pos_idx])
            p_neg = sum([self.gmm_pdf[j](cf) for j in neg_idx])
            # p = sum([self.gmm_pdf[j](cf) for j in range(len(self.gmm_pdf))])
            if uniform(0.0, p_pos) > -1. * p_neg:
                accepted = True
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
        mfcc_norm = mfcc / self.norm_coef
        # calc average distance to all units in test_data
        distance = np.linalg.norm(mfcc_norm - self.target)
        # calculate goodness of exploration: sigmoid with zero  at distance 3.0
        goodness = 2.0 / (1.0 + np.exp(5.0*(distance - 0.6))) - 1.
        # assign weight for new gmm component
        w = goodness
        center = cf
        covariance = np.identity(self.num_params) * (0.02 + 0.01)
        new_comp = lambda c=center, cov=covariance: np.random.multivariate_normal(c, cov)

        self.gmm_weights.append(w)
        self.gmm.append(new_comp)
        self.gmm_pdf.append(lambda x, c=center, cov=covariance: multivariate_normal.pdf(x, c, cov))
        print "distance:", distance
        print "weight: ", np.around(self.gmm_weights[-1], decimals=2)

        return distance, mfcc_norm, w, center, covariance

