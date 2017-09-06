import random
import numpy as np
import VTL_API.pyvtl_v1 as pyvtl

print pyvtl.tract_param_names.value.decode().split()
print pyvtl.glottis_param_names.value.decode().split()


def explore_articulatory_space(cf, sigma, targets=20, fixed_params=[]):
    """Explores articulatory space.

    :param cf: the vocal tract configuration to explore from
    :param sigma: sigma for normal distribution
    :param k_targets: number of new targets
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


def synthesise_static_sound(cf, duration=0.5, frame_rate=200):
    num_frames = int(duration*frame_rate)
    tract_params = np.tile(cf[:24], (num_frames, 1))
    glottis_params = np.tile(cf[24:], (num_frames, 1))
    audio = pyvtl.synth_block(tract_params, glottis_params, frame_rate)
    return audio


def test_0():
    test_cf = pyvtl.tract_param_neutral + pyvtl.glottis_param_neutral
    max_params = pyvtl.tract_param_max + pyvtl.glottis_param_max
    min_params = pyvtl.tract_param_min + pyvtl.glottis_param_min
    test_cf = [(test_cf[i] - min_params[i]) / (max_params[i] - min_params[i])
               for i in range(len(test_cf))]
    fixed_params = [1, 2, 3]
    print "initial_state: ", test_cf
    print "explored states: ", explore_articulatory_space(test_cf, 0.5, 3, fixed_params)
    return


def test_1():
    from scipy.io import wavfile
    test_cf = pyvtl.tract_param_neutral + pyvtl.glottis_param_neutral
    max_params = pyvtl.tract_param_max + pyvtl.glottis_param_max
    min_params = pyvtl.tract_param_min + pyvtl.glottis_param_min
    test_cf = [(test_cf[i] - min_params[i]) / (max_params[i] - min_params[i])
               for i in range(len(test_cf))]

    audio = synthesise_static_sound(test_cf)
    wav = np.array(audio)
    wav_int = np.int16(wav * (2 ** 15 - 1))
    wavfile.write('ai_test.wav', 22050, wav_int)
    return


def test_2():
    from scipy.io import wavfile
    test_cf = pyvtl.tract_param_neutral + pyvtl.glottis_param_neutral
    max_params = pyvtl.tract_param_max + pyvtl.glottis_param_max
    min_params = pyvtl.tract_param_min + pyvtl.glottis_param_min
    test_cf = [(test_cf[i] - min_params[i]) / (max_params[i] - min_params[i])
               for i in range(len(test_cf))]
    fixed_params = range(24, 30)
    new_cfs = explore_articulatory_space(test_cf, 0.05, 5, fixed_params)
    for i in range(len(new_cfs)):
        audio = synthesise_static_sound(new_cfs[i])
        wav = np.array(audio)
        wav_int = np.int16(wav * (2 ** 15 - 1))
        wavfile.write('ai_test_{0}.wav'.format(i), 22050, wav_int)

test_2()
