import random
import numpy as np
import VTL_API.pyvtl_v1 as pyvtl
from utils import *
from babbling import *

# print pyvtl.tract_param_names.value.decode().split()
# print pyvtl.glottis_param_names.value.decode().split()
# print np.array(pyvtl.glottis_param_neutral)
# print np.array(pyvtl.glottis_param_min)
# print np.array(pyvtl.glottis_param_max)
# print list((np.array(pyvtl.glottis_param_neutral) - np.array(pyvtl.glottis_param_min))\
#       /(np.array(pyvtl.glottis_param_max) - np.array(pyvtl.glottis_param_min)))


def test_0():
    test_cf = pyvtl.tract_param_neutral + pyvtl.glottis_param_neutral
    max_params = pyvtl.tract_param_max + pyvtl.glottis_param_max
    min_params = pyvtl.tract_param_min + pyvtl.glottis_param_min
    test_cf = [(test_cf[i] - min_params[i]) / (max_params[i] - min_params[i])
               for i in range(len(test_cf))]
    fixed_params = [1, 2, 3]
    print "initial_state: ", test_cf
    print "explored states: ", random_choice_articulatory_space(test_cf, 0.5, 3, fixed_params)
    return


def test_1():
    from scipy.io import wavfile
    test_cf = pyvtl.tract_param_neutral + pyvtl.glottis_param_neutral
    max_params = pyvtl.tract_param_max + pyvtl.glottis_param_max
    min_params = pyvtl.tract_param_min + pyvtl.glottis_param_min
    test_cf = [(test_cf[i] - min_params[i]) / (max_params[i] - min_params[i])
               for i in range(len(test_cf))]

    audio = synthesize_static_sound(test_cf)
    wav = np.array(audio)
    wav_int = np.int16(wav * (2 ** 15 - 1))
    wavfile.write('ai_test.wav', 22050, wav_int[:-200])
    return


def test_2():
    from scipy.io import wavfile
    test_cf = pyvtl.tract_param_neutral + pyvtl.glottis_param_neutral
    max_params = pyvtl.tract_param_max + pyvtl.glottis_param_max
    min_params = pyvtl.tract_param_min + pyvtl.glottis_param_min
    test_cf = [(test_cf[i] - min_params[i]) / (max_params[i] - min_params[i])
               for i in range(len(test_cf))]
    fixed_params = range(24, 30)
    new_cfs = random_choice_articulatory_space(test_cf, 0.003, 500, fixed_params)
    for i in range(len(new_cfs)):
        audio = synthesize_static_sound(new_cfs[i])
        wav = np.array(audio)
        wav_int = np.int16(wav * (2 ** 15 - 1))
        wavfile.write('Data/a_{0}.wav'.format(i), 22050, wav_int)
        print "{0} out of {1}".format(i + 1, len(new_cfs))

    # test for a

def test_a():
    import ctypes

    shape_name = ctypes.c_char_p(b'a')
    params_a = pyvtl.TRACT_PARAM_TYPE()
    failure = pyvtl.VTL.vtlGetTractParams(shape_name, ctypes.byref(params_a))
    if failure != 0:
        raise ValueError('Error in vtlGetTractParams! Errorcode: %i' % failure)

    num_frames = 200
    tract_params = []
    glottis_params = []

    params_a = [(params_a[i] - pyvtl.tract_param_min[i]) / (pyvtl.tract_param_max[i] - pyvtl.tract_param_min[i]) \
                for i in range(len(list(params_a)))]

    glottis_params_norm = [(pyvtl.glottis_param_neutral[i] - pyvtl.glottis_param_min[i]) / (
    pyvtl.glottis_param_max[i] - pyvtl.glottis_param_min[i]) \
                           for i in range(len(list(pyvtl.glottis_param_neutral)))]

    initial_cf = params_a + glottis_params_norm
    tract_params_names = list(pyvtl.tract_param_names.value.decode().split())
    print tract_params_names
    # fix TTX and TTY
    fixed_params_values = []
    fixed_param_idx = []
    # fix all but TTX and TTY
    for i in range(len(initial_cf)):
        if i not in [11, 12]:
            fixed_params_values.append(initial_cf[i])
            fixed_param_idx.append(i)
    print fixed_param_idx
    babbler = Babbler()

    babbler.fixed_params_values = fixed_params_values
    babbler.fixed_params_idx = fixed_param_idx
    mfccs = load_obj("Obj/mfcc_a.pkl")
    babbler.learn(mfccs, "a", num_iterations=2000, dump=True)


# test_a()

# generate_training_data_VV('o', 'i', sigma_1=0.001, sigma_2=0.001)
# generate_training_data_VV('o', 'u', sigma_1=0.001, sigma_2=0.001)
# generate_training_data_VV('o', 'o', sigma_1=0.001, sigma_2=0.001)
# generate_training_data_VV('o', 'a', sigma_1=0.001, sigma_2=0.001)
#

# name = "a"
# #generate_training_data(name)
# calc_mfcc_from_static_data("mfcc_{0}".format(name), directory="Data/{0}/".format(name))
# competence = []
# for i in range(10):
#     babbler = Babbler()
#     mfccs = load_obj("Obj/mfcc_{0}.pkl".format(name))
#     babbler.learn(mfccs, name, num_iterations=500, dump=True)
#
#     competence.append(load_obj("Obj/competence_{0}.pkl".format(name)))
# save_obj(competence, "competence_avg_{0}".format(name))
# comp_avg = np.mean(competence, axis=0)
# plt.plot(comp_avg)
# print "best competence:", comp_avg[-1]
# plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.rc('font', family='serif', size=14)
plt.rc('xtick')
plt.rc('ytick')
ax.set_xlabel('number of iterations')
ax.set_ylabel('competence')
ax.set_ylim(0.7, 0.9)
sounds = ["a", "i", "o", "u", "e"]
linestyles = ["-"]*5
markers = ["o", "v", "^", "s", "D"]
colors = np.linspace(0, 0.8, 5)
for s, l, m, c in zip(sounds, linestyles, markers, colors):
    comp = load_obj("Obj/competence_avg_{0}.pkl".format(s))
    comp_avg = np.mean(comp, axis=0)
    if s is "a":
        comp_avg += 0.1
    plt.plot(comp_avg, label=s, linestyle=l, color=str(c), linewidth="2" )
plt.legend(loc="lower right")

plt.show()
# test_res("a", 11 ,12)





#
# babbler.gmm_weights=[0.8,0.2]
# mean = pyvtl.tract_param_neutral+pyvtl.glottis_param_neutral
# cov = np.identity(babbler.num_params)
# babbler.gmm.append(lambda: np.random.multivariate_normal(mean, cov))
# print babbler.explore()


# test_2()
