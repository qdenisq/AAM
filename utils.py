import os
import pickle
import scipy.io.wavfile as wav
from scipy.io import wavfile
import random
import numpy as np
import matplotlib.pyplot as plt
import speechpy
# import librosa
from scipy.stats import multivariate_normal

from mpl_toolkits.mplot3d import Axes3D

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


def synthesize_dynamic_sound(cf_1, cf_2, duration=1.0, frame_rate=200):
    num_frames = int(duration*frame_rate)
    tract_params = []
    glottis_params = []
    for i in range(int(num_frames / 3)):
        tract_params.append([cf_1[p] for p in range(24)])
        glottis_params.append([cf_1[p] for p in range(24, 30)])
    for i in range(int(num_frames/3), int(2.*num_frames/3)):
        tract_params.append([cf_1[p] + (cf_2[p] - cf_1[p]) * (3. * i - num_frames) / num_frames for p in range(24)])
        glottis_params.append([cf_1[p] + (cf_2[p] - cf_1[p]) * (3. * i - num_frames) / num_frames for p in range(24, 30)])
    for i in range(int(2*num_frames / 3), num_frames):
        tract_params.append([cf_2[p] for p in range(24)])
        glottis_params.append([cf_2[p] for p in range(24, 30)])
    audio = pyvtl.synth_block(tract_params, glottis_params, frame_rate)
    return audio


def create_static_sound_data(cf, name, sigma=0.003, num_samples=500, folder="Data"):
    fixed_params = range(24, 30)
    new_cfs = random_choice_articulatory_space(cf, sigma, num_samples, fixed_params)
    for i in range(len(new_cfs)):
        audio = synthesize_static_sound(new_cfs[i])
        wav = np.array(audio)
        wav_int = np.int16(wav * (2 ** 15 - 1))
        wavfile.write('{0}/{1}_{2}.wav'.format(folder, name, i), 22050, wav_int)
        print "{0} out of {1}".format(i + 1, len(new_cfs))
    return


def create_dynamic_sound_data(cf_1, cf_2, name, sigma_1=0.003, sigma_2=0.003, num_samples=500, folder="Data", duration=0.5):
    fixed_params = range(24, 30)
    cfs_1 = random_choice_articulatory_space(cf_1, sigma_1, num_samples, fixed_params)
    cfs_2 = random_choice_articulatory_space(cf_2, sigma_2, num_samples, fixed_params)

    for i in range(len(cfs_1)):
        audio = synthesize_dynamic_sound(cfs_1[i], cfs_2[i], duration=duration)
        wav = np.array(audio)
        wav_int = np.int16(wav * (2 ** 15 - 1))
        wavfile.write('{0}/{1}_{2}.wav'.format(folder, name, i), 22050, wav_int)
        print "{0} out of {1}".format(i + 1, len(cfs_1))
    return


def calc_mfcc_from_vowel(signal, fs=22050):
    mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                         num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    n = len(mfcc)
    mean_mfcc = np.mean(mfcc[n/4:n*3/4], axis=0)
    return mean_mfcc


def calc_mfcc_from_static_data(directory):
    mfccs = []
    fnames = get_file_list(directory)
    for fname in fnames:
        fs, signal = wav.read(os.path.join(directory, fname))
        mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                         num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
        n = len(mfcc)
        mean_mfcc = np.mean(mfcc[n / 4:n * 3 / 4], axis=0)
        mfccs.append(mean_mfcc)
    return mfccs


def calc_mfcc_from_dynamic_data(directory):
    mfccs = []
    mfccs_d = []
    fnames = get_file_list(directory)
    for fname in fnames:
        fs, signal = wav.read(os.path.join(directory, fname))
        mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                         num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
        mfccs.append(mfcc)
        mfcc_d = speechpy.extract_derivative_feature(mfcc)
        mfccs_d.append(mfcc_d)
    return mfccs, mfccs_d


def extract_mfcc(directory):
    mfccs_d = []
    fnames = get_file_list(directory)
    for fname in fnames:
        fs, signal = wav.read(os.path.join(directory, fname))
        mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
        mfcc_d = speechpy.extract_derivative_feature(mfcc)
        mfccs_d.append(mfcc_d)
    return mfccs_d


def plot_gmm_3d(components_list, axes_names, axes_idx, targets=[], best_match=[]):
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    rvs = []
    values = np.zeros((len(x), len(y)))
    for i in range(len(components_list)):
        weight, center, covariance = components_list[i]
        c = [center[axes_idx[0]], center[axes_idx[1]]]
        cov = np.max(covariance) * np.identity(2)
        rvs.append(multivariate_normal(c, cov))
        values += weight * rvs[-1].pdf(pos)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, values, cmap='viridis', linewidth=0)

    tx, ty = [list(t) for t in zip(*targets)]
    best_x, best_y = [list(b) for b in zip(*best_match)]
    ax.scatter(tx, ty, np.max(values), c="red", marker=">", s=100)
    ax.scatter(best_x, best_y, np.max(values), c="green", marker="o", s=100)

    plt.show()


def test_res(sound, p0, p1):
    print pyvtl.tract_param_names.value.decode()

    comps = load_obj("Obj/components_{0}.pkl".format(sound))
    import ctypes
    shape_name = ctypes.c_char_p(sound)
    params_a = pyvtl.TRACT_PARAM_TYPE()
    failure = pyvtl.VTL.vtlGetTractParams(shape_name, ctypes.byref(params_a))
    if failure != 0:
        raise ValueError('Error in vtlGetTractParams! Errorcode: %i' % failure)

    params_a = [(params_a[i] - pyvtl.tract_param_min[i]) / (pyvtl.tract_param_max[i] - pyvtl.tract_param_min[i]) \
                for i in range(len(list(params_a)))]

    glottis_params_norm = [(pyvtl.glottis_param_neutral[i] - pyvtl.glottis_param_min[i]) / (pyvtl.glottis_param_max[i] - pyvtl.glottis_param_min[i]) \
                           for i in range(len(list(pyvtl.glottis_param_neutral)))]
    target = [(params_a[p0], params_a[p1])]
    initial_cf = params_a + glottis_params_norm

    best_match_cf = load_obj("Obj/best_cf_{0}.pkl".format(sound))
    best_match = [(best_match_cf[p0], best_match_cf[p1])]
    print "target: ", target
    print "best match: ",best_match
    print np.round(np.array(best_match_cf)-np.array(initial_cf), decimals=2)
    print "distance between target and result: ", np.linalg.norm(np.array(best_match)-np.array(target))

    plot_gmm_3d(comps, ["TBX", "TBY"], [p0, p1], target, best_match)


def generate_training_data(sound, sigma=0.001):
    import ctypes
    shape_name = ctypes.c_char_p(sound)
    params = pyvtl.TRACT_PARAM_TYPE()
    failure = pyvtl.VTL.vtlGetTractParams(shape_name, ctypes.byref(params))
    if failure != 0:
        raise ValueError('Error in vtlGetTractParams! Errorcode: %i' % failure)

    params = [(params[i] - pyvtl.tract_param_min[i]) / (pyvtl.tract_param_max[i] - pyvtl.tract_param_min[i]) \
                for i in range(len(list(params)))]

    glottis_params_norm = [(pyvtl.glottis_param_neutral[i] - pyvtl.glottis_param_min[i]) / (
    pyvtl.glottis_param_max[i] - pyvtl.glottis_param_min[i]) \
                           for i in range(len(list(pyvtl.glottis_param_neutral)))]

    cf = params + glottis_params_norm

    create_static_sound_data(cf, sigma=sigma, folder="Data/{0}".format(sound), name=sound)
    mfccs = calc_mfcc_from_static_data("Data/{0}".format(sound))
    save_obj(mfccs, "mfcc_{0}".format(sound), directory="Obj")
    return


def generate_training_data_VV(sound_1, sound_2, sigma_1=0.001, sigma_2=0.001):
    import ctypes
    shape_name_1 = ctypes.c_char_p(sound_1)
    params_1 = pyvtl.TRACT_PARAM_TYPE()
    failure = pyvtl.VTL.vtlGetTractParams(shape_name_1, ctypes.byref(params_1))
    if failure != 0:
        raise ValueError('Error in vtlGetTractParams! Errorcode: %i' % failure)

    shape_name_2 = ctypes.c_char_p(sound_2)
    params_2 = pyvtl.TRACT_PARAM_TYPE()
    failure = pyvtl.VTL.vtlGetTractParams(shape_name_2, ctypes.byref(params_2))
    if failure != 0:
        raise ValueError('Error in vtlGetTractParams! Errorcode: %i' % failure)

    params_1 = [(params_1[i] - pyvtl.tract_param_min[i]) / (pyvtl.tract_param_max[i] - pyvtl.tract_param_min[i]) \
                for i in range(len(list(params_1)))]

    params_2 = [(params_2[i] - pyvtl.tract_param_min[i]) / (pyvtl.tract_param_max[i] - pyvtl.tract_param_min[i]) \
                for i in range(len(list(params_2)))]

    glottis_params_norm = [(pyvtl.glottis_param_neutral[i] - pyvtl.glottis_param_min[i]) / (
        pyvtl.glottis_param_max[i] - pyvtl.glottis_param_min[i]) \
                           for i in range(len(list(pyvtl.glottis_param_neutral)))]

    cf_1 = params_1 + glottis_params_norm
    cf_2 = params_2 + glottis_params_norm

    name = sound_1 + sound_2
    print name
    create_dynamic_sound_data(cf_1, cf_2, name=name, sigma_1=sigma_1, sigma_2=sigma_2,
                              folder="Data/{0}{1}".format(sound_1, sound_2), num_samples=10, duration=0.5)
    mfccs, _ = calc_mfcc_from_dynamic_data("Data/{0}{1}".format(sound_1, sound_2))
    save_obj(mfccs, "mfcc_{0}{1}".format(sound_1, sound_2), directory="Obj")
    return


def normalize_vector(vec):
    max_v = vec.max()
    min_v = vec.min()
    return [(v - min_v)/(max_v - min_v) for v in vec]


def normalize(data, axis=0, norm_by_column=False):
    if norm_by_column:
        col_maxes = data.max(axis=axis)
        col_mins = data.min(axis=axis)

        normed_data = np.apply_along_axis(normalize_vector, axis, data)
        return normed_data, col_maxes, col_mins
    else:
        max_v = data.max()
        min_v = data.min()

        normed_data = np.true_divide(np.subtract(data, min_v), max_v - min_v)
        return normed_data, max_v, min_v


def timit_calc_mfcc(directory):
    mfccs, mfccs_cube = calc_mfcc_from_dynamic_data(directory)

    data = np.vstack(mfccs_cube)

    normed_data = normalize(data, axis=0, norm_by_column=True)

    save_obj(normed_data, "timit_mfccs")

    return mfccs_cube


def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape
    N_pad = N_tar - shape[0]
    print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append
    if shape[0] > 0:
        if len(shape) > 1:
            return np.vstack((data, np.zeros(shape)))
        else:
            return np.hstack((data, np.zeros(shape)))
    else:
        return data

generate_training_data_VV("a", "i")

#
# directory = r"C:\Study\DB\USC-TIMIT\USC-TIMIT\EMA\Data\M1\wav"
# timit_calc_mfcc(directory)

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

