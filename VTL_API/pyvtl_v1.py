#!/usr/bin/env python3

"""
This example generates the transition from /a/ to /i/ using the vocal tract
model and the function vtlSynthBlock(...).

.. note::

    This example uses ``ctypes`` and is very close to the vocaltractlab API.
    This comes with breaking with some standard assumptions one have in python.
    If one wants a more pythonic experience use the ``pyvtl`` wrapper in order
    to use vocaltractlab from python. (pyvtl does not exist yet)

If you are not aware of ``ctypes`` read the following introduction
https://docs.python.org/3/library/ctypes.html

For an in-depth API description look at the `VocalTractLabApi64.h`.

For plotting and saving results you need to install ``matplotlib``, ``numpy``,
and ``scipy``.

"""
import os
import ctypes
import sys

# try to load some non-essential packages
try:
    import numpy as np
except ImportError:
    np = None
try:
    from scipy.io import wavfile
except ImportError:
    wavefile = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# load vocaltractlab binary
# Use 'VocalTractLabApi32.dll' if you use a 32-bit python version.
if sys.platform == 'win32':
    VTL = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VocalTractLabApi64.dll'))
else:
    VTL = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)),'VocalTractLabApi64.so'))


# get version / compile date
version = ctypes.c_char_p(b'                                ')
VTL.vtlGetVersion(version)
# print('Compile date of the library: "%s"' % version.value.decode())


# initialize vtl
speaker_file_name = ctypes.c_char_p(os.path.join(os.path.dirname(os.path.abspath(__file__)),'JD2.speaker').encode())

failure = VTL.vtlInitialize(speaker_file_name)
if failure != 0:
    raise ValueError('Error in vtlInitialize! Errorcode: %i' % failure)


# get some constants
audio_sampling_rate = ctypes.c_int(0)
number_tube_sections = ctypes.c_int(0)
number_vocal_tract_parameters = ctypes.c_int(0)
number_glottis_parameters = ctypes.c_int(0)

VTL.vtlGetConstants(ctypes.byref(audio_sampling_rate),
                    ctypes.byref(number_tube_sections),
                    ctypes.byref(number_vocal_tract_parameters),
                    ctypes.byref(number_glottis_parameters))

# print('Audio sampling rate = %i' % audio_sampling_rate.value)
# print('Num. of tube sections = %i' % number_tube_sections.value)
# print('Num. of vocal tract parameters = %i' % number_vocal_tract_parameters.value)
# print('Num. of glottis parameters = %i' % number_glottis_parameters.value)


# get information about the parameters of the vocal tract model
# Hint: Reserve 32 chars for each parameter.
TRACT_PARAM_TYPE = ctypes.c_double * number_vocal_tract_parameters.value
tract_param_names = ctypes.c_char_p((' ' * 32 * number_vocal_tract_parameters.value).encode())
tract_param_min = TRACT_PARAM_TYPE()
tract_param_max = TRACT_PARAM_TYPE()
tract_param_neutral = TRACT_PARAM_TYPE()

VTL.vtlGetTractParamInfo(tract_param_names,
                         ctypes.byref(tract_param_min),
                         ctypes.byref(tract_param_max),
                         ctypes.byref(tract_param_neutral))

tract_param_max = list(tract_param_max)
tract_param_min = list(tract_param_min)

# print('Vocal tract parameters: "%s"' % tract_param_names.value.decode())
# print('Vocal tract parameter minima: ' + str(list(tract_param_min)))
# print('Vocal tract parameter maxima: ' + str(list(tract_param_max)))
# print('Vocal tract parameter neutral: ' + str(list(tract_param_neutral)))

# get information about the parameters of glottis model
# Hint: Reserve 32 chars for each parameter.
GLOTTIS_PARAM_TYPE = ctypes.c_double * number_glottis_parameters.value
glottis_param_names = ctypes.c_char_p((' ' * 32 * number_glottis_parameters.value).encode())
glottis_param_min = GLOTTIS_PARAM_TYPE()
glottis_param_max = GLOTTIS_PARAM_TYPE()
glottis_param_neutral = GLOTTIS_PARAM_TYPE()

VTL.vtlGetGlottisParamInfo(glottis_param_names,
                           ctypes.byref(glottis_param_min),
                           ctypes.byref(glottis_param_max),
                           ctypes.byref(glottis_param_neutral))

glottis_param_max = list(glottis_param_max)
glottis_param_min = list(glottis_param_min)

"""
input:
    - tract_params : list of lists of vocal tract parameters in range [0.0 : 1.0]
    - glottis_params : list of lists of glottis parameters
    - duration : duration for audio output in seconds

output:
    - audio : lits of doubles in range [-1.0 : 1.0]
"""


def synth_block(tract_params_norm, glottis_params_norm, frame_rate = 200.0):
    if len(tract_params_norm) != len(glottis_params_norm):
        print('number of tract and glottis parameters don\'t match : %s ; %s',
              len(tract_params_norm), len(glottis_params_norm))
        return None

    number_frames = len(tract_params_norm)
    duration = float(number_frames) / frame_rate

    audio = (ctypes.c_double * int(duration * audio_sampling_rate.value + 2000))()
    number_audio_samples = ctypes.c_int(0)

    # init the arrays
    tract_params = (ctypes.c_double * (number_frames * number_vocal_tract_parameters.value))()
    glottis_params = (ctypes.c_double * (number_frames * number_glottis_parameters.value))()

    tube_areas = (ctypes.c_double * (number_frames * number_tube_sections.value))()
    tube_articulators = ctypes.c_char_p(b' ' * number_frames * number_tube_sections.value)

    # Create the vocal tract shapes that slowly change from /a/ to /i/ from the
    # first to the last frame.

    for ii in range(number_frames):
        for jj in range(number_vocal_tract_parameters.value):
            tract_params[ii * number_vocal_tract_parameters.value + jj] = \
                tract_params_norm[ii][jj] * (tract_param_max[jj] - tract_param_min[jj]) + tract_param_min[jj]

        offset_glottis = ii * number_glottis_parameters.value
        # transition F0 in Hz going from 120 Hz down to 100 Hz
        glottis_params[offset_glottis + 0] = 120.0 - 20.0 * (ii / number_frames)

        # Start with zero subglottal pressure and then go to 1000 Pa.
        # Somehow, P_sub must stay at zero for the first two frames - otherwise
        # we get an annoying transient at the beginning of the audio signal,
        # and I (Peter) don't know why this is so at the moment.
        if ii <= 1:
            glottis_params[offset_glottis + 1] = 0.0
        elif ii == 2:
            glottis_params[offset_glottis + 1] = 500.0
        else:
            glottis_params[offset_glottis + 1] = 1000.0

        # use the neutral settings for the rest of glottis parameters
        for jj in range(2, number_glottis_parameters.value):
            glottis_params[offset_glottis + jj] = \
                glottis_params_norm[ii][jj] * (glottis_param_max[jj] - glottis_param_min[jj]) + glottis_param_min[jj]

    # Call the synthesis function. It may calculate a few seconds.
    failure = VTL.vtlSynthBlock(ctypes.byref(tract_params),  # input
                                ctypes.byref(glottis_params),  # input
                                ctypes.byref(tube_areas),  # output
                                tube_articulators,  # output
                                number_frames,  # input
                                ctypes.c_double(frame_rate),  # input
                                ctypes.byref(audio),  # output
                                ctypes.byref(number_audio_samples))  # output

    if failure != 0:
        raise ValueError('Error in vtlSynthBlock! Errorcode: %i' % failure)

    # destroy current state of VTL and free memory
    VTL.vtlClose()

    return audio


def test():
    import random
    import time
    shape_name = ctypes.c_char_p(b'i')
    params_a = TRACT_PARAM_TYPE()
    failure = VTL.vtlGetTractParams(shape_name, ctypes.byref(params_a))
    if failure != 0:
        raise ValueError('Error in vtlGetTractParams! Errorcode: %i' % failure)

    num_frames = 200
    tract_params = []
    glottis_params = []

    params_a = [(params_a[i] - tract_param_min[i])/(tract_param_max[i] - tract_param_min[i])\
                for i in range(len(list(params_a)))]

    glottis_params_norm = [(glottis_param_neutral[i] - glottis_param_min[i])/(glottis_param_max[i] - glottis_param_min[i])\
                for i in range(len(list(glottis_param_neutral)))]

    for i in range(num_frames):
        tract_params.append([min(p + random.random()/50.0, 1.0)for p in params_a])
        glottis_params.append([min(p + random.random()/50.0, 1.0)for p in glottis_params_norm])

    time_start = time.clock()

    audio = synth_block(tract_params, glottis_params)

    print("time_elapsed: ", time.clock() - time_start)

    # plot and save the audio signal
    ################################

    if np is not None:
        wav = np.array(audio)

    if plt is not None and np is not None:
        frame_rate = 200.0
        time_wav = np.arange(len(wav)) / frame_rate
        plt.plot(time_wav, wav, c='black', alpha=0.75)
        plt.ylabel('amplitude')
        plt.ylim(-1, 1)
        plt.xlabel('time [s]')
        print('\nClose the plot in order to continue.')
        plt.show()
    else:
        print('plotting not available; matplotlib needed')
        print('skip plotting')

    if wavfile is not None and np is not None:
        wav_int = np.int16(wav * (2 ** 15 - 1))
        wavfile.write('ai_test.wav', audio_sampling_rate.value, wav_int)
        print('saved audio to "ai_test.wav"')
    else:
        print('scipy not available')
        print('skip writing out wav file')
    return


def test1():
    import scipy.io.wavfile as wav
    import numpy as np
    import speechpy
    import librosa

    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usctimit_ema_m1_001_005.wav')
    fs, signal = wav.read(file_name)

    ############# Extract MFCC features #############
    mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                         num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    mfcc_feature_cube = speechpy.extract_derivative_feature(mfcc)
    plt.subplot(2,1,1)
    plt.imshow( np.transpose(mfcc), cmap='hot', interpolation='nearest',aspect='auto',
               origin='lower')
    print('mfcc feature cube shape=', mfcc_feature_cube.shape)
    plt.colorbar()

    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13, n_fft=512, hop_length=220, power=1.0)
    plt.subplot(2,1,2)
    plt.imshow(mfcc, cmap='hot', interpolation='nearest', aspect='auto',
               origin='lower')
    plt.colorbar()
    plt.show()

# test()
# test1()