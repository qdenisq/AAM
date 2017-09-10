import os
import pickle
import scipy.io.wavfile as wav
from scipy.io import wavfile
import random
import numpy as np
import matplotlib.pyplot as plt
import speechpy
import librosa

import VTL_API.pyvtl_v1 as pyvtl


def save_obj(obj, name, directory=''):
    if directory == '':
        directory = r"obj/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, name+'.pkl'), 'wb') as f:
        pickle.dump(obj, f, protocol=0)


def load_obj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_file_list(directory, ext='.wav'):
    return [fname for fname in os.listdir(directory) if fname.endswith(ext)]


def random_choice_articulatory_space(cf, sigma, targets=20, fixed_params=[]):
    """Explores articulatory space.

    :param cf: the vocal tract configuration to explore from
    :param sigma: sigma for normal distribution
    :param targets: number of new targets
    :param fixed_params: list of fixed parameters (these parameters will same as in cf)
    :return new_configurations: list of explored configurations
    """
    new_congifurations = []
    for i in range(targets):
        configuration = [random.normalvariate(cf[i], sigma) if i not in fixed_params else cf[i]
                         for i in range(len(cf))]
        configuration = [min(max(p, 0.0), 1.0) for p in configuration]
        new_congifurations.append(configuration)
    return new_congifurations


def synthesize_static_sound(cf, duration=0.5, frame_rate=200):
    num_frames = int(duration*frame_rate)
    tract_params = np.tile(cf[:24], (num_frames, 1))
    glottis_params = np.tile(cf[24:], (num_frames, 1))
    audio = pyvtl.synth_block(tract_params, glottis_params, frame_rate)
    return audio


def create_static_sound_data(cf, sigma=0.003, num_samples=500, folder="Data"):

    test_cf = pyvtl.tract_param_neutral + pyvtl.glottis_param_neutral
    max_params = pyvtl.tract_param_max + pyvtl.glottis_param_max
    min_params = pyvtl.tract_param_min + pyvtl.glottis_param_min
    test_cf = [(test_cf[i] - min_params[i]) / (max_params[i] - min_params[i])
               for i in range(len(test_cf))]
    fixed_params = range(24, 30)
    new_cfs = random_choice_articulatory_space(test_cf, sigma, num_samples, fixed_params)
    for i in range(len(new_cfs)):
        audio = synthesize_static_sound(new_cfs[i])
        wav = np.array(audio)
        wav_int = np.int16(wav * (2 ** 15 - 1))
        wavfile.write('{0}/a_{1}.wav'.format(folder, i), 22050, wav_int)
        print "{0} out of {1}".format(i + 1, len(new_cfs))
    return


def calc_mfcc_from_vowel(signal, fs=22050):

    mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                         num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    n = len(mfcc)
    mean_mfcc = np.mean(mfcc[n/4:n*3/4], axis=0)
    return mean_mfcc


def calc_mfcc_from_static_data(directory='Data'):
    fnames = get_file_list(directory)
    mfccs = [calc_mfcc_from_vowel(wav.read(os.path.join(directory, file_name))[1]) for file_name in fnames]
    print len(mfccs)
    save_obj(mfccs, "mfccs_a", directory="Obj")
    return mfccs


#
#
# a = np.zeros((3, 3), float)
# print np.fill_diagonal(a, 1.0)
# print a
#
#
# from scipy.stats import norm
#
# x = np.linspace(0,1,100)
#
# sigma=0.1
#
# # 5 components
# means = [0.5, 0.3, 0.5, 0.8, 0.9]
# sigmas = [0.02, 0.2, 0.1, 0.05, 0.1]
# weights = [-0.2, -0.5, 0.3, 0.4, 0.3, 0.7]
#
# pdfs = [weights[i]*norm.pdf(x, loc=means[i], scale=sigmas[i]) for i in xrange(len(means))]
# plt.subplot(2,1,1)
# for pdf in pdfs:
#     plt.plot(x, pdf, linestyle="--")
# uniform = [weights[-1]]*len(x)
# plt.plot(x, uniform, linestyle="--")
# p = sum(pdfs)+uniform
# plt.plot(x, p, linewidth="2")
#
# plt.subplot(2,1,2)
# weights = [w+min(weights)*-1.0 for w in weights]
# weights = [w/sum(weights) for w in weights]
#
# pdfs = [weights[i]*norm.pdf(x, loc=means[i], scale=sigmas[i]) for i in xrange(len(means))]
# plt.subplot(2,1,2)
# for pdf in pdfs:
#     plt.plot(x, pdf, linestyle="--")
# uniform = [weights[-1]]*len(x)
# plt.plot(x, uniform, linestyle="--")
# p = sum(pdfs)+uniform
# plt.plot(x, p, linewidth="2")
#
# plt.show()




#
#
# mfccs = calc_mfcc_from_static_data("Data")
# plt.imshow(np.transpose(mfccs), cmap='hot', interpolation='nearest',aspect='auto',
#                origin='lower')
# plt.show()




# mfccs = load_obj("Obj/mfccs_a.pkl")
# print len(mfccs)
#
# distance = []
# for i in range(500):
#     for j in range(i, 500):
#         distance.append(np.linalg.norm(np.array(mfccs[i])-np.array(mfccs[j])))
# plt.hist(distance)
# plt.show()





#
# mu = 0
# sigma = 0.1
# s= np.random.normal(mu, sigma, 1000)
# plt.hist(s)
# plt.show()



#
# cf = load_obj("Obj/cf_a.pkl")
# synthesize_static_sound(cf)
# audio = synthesize_static_sound(cf)
# wav = np.array(audio)
# wav_int = np.int16(wav * (2 ** 15 - 1))
# wavfile.write('Obj/a.wav', 22050, wav_int)
