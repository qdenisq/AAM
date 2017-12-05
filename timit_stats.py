import os
import scipy.io as sio
# import gesture as ges
import math
# import critical_point as cp
import cPickle
import numpy as np
from scipy import interpolate
import scipy.io.wavfile as wav
from scipy.io import wavfile

#
# trans_dir = "../USC-TIMIT/EMA/Data/M1/trans"
# mat_dir = "../USC-TIMIT/EMA/Data/M1/mat"
# trans = os.listdir(trans_dir)


def save_obj(obj, name):
    if not os.path.exists('obj/'):
        os.makedirs("obj")
    with open('obj/' + name + '.pkl', 'wb') as f:
        cPickle.dump(obj, f, cPickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return cPickle.load(f)


def list_TIMIT_dir(root_dir):
    trans_dir = os.path.join(root_dir, "trans")
    mat_dir = os.path.join(root_dir, "mat")
    wav_dir = os.path.join(root_dir, "wav")
    return [(os.path.join(trans_dir, fname),
              os.path.join(mat_dir, os.path.splitext(fname)[0] + ".mat"),
             os.path.join(wav_dir, os.path.splitext(fname)[0] + ".wav")) for fname in os.listdir(trans_dir)]


def parse_transcription(fname):
    t_starts = []
    t_ends = []
    phonemes = []
    words = []
    sentences = []
    with open(fname) as f:
        content = f.readlines()
    for line in content:
        word_list = line.split(',')
        t_starts.append(float(word_list[0]))
        t_ends.append(float(word_list[1]))
        phonemes.append(word_list[2])
        words.append(word_list[3])
        sentences.append(word_list[4])
    return t_starts, t_ends, phonemes, words, sentences


def parse_trans_corpus(dir):
    t_starts = []
    t_ends = []
    phonemes = []
    words = []
    sentences = []
    for fname in trans:
        t_s, t_e, ph, w, s = parse_transcription(dir + '/' + fname)
        t_starts.extend(t_s)
        t_ends.extend(t_e)
        phonemes.extend(ph)
        words.extend(w)
        sentences.extend(s)
    return t_starts, t_ends, phonemes, words, sentences


def parse_mat(mat_fname):
    # load mat file
    mat_contents = sio.loadmat(mat_fname)

    name = os.path.splitext(os.path.basename(mat_fname))[0]
    data = mat_contents[name]
    rows = len(data[0])
    params = {}
    srates = {}
    for r in range(1,rows):
        p = data[0,r][0][0]
        params[p + "_x"] = []
        params[p + "_y"] = []
        params[p + "_z"] = []
        srates[p + "_x"] = data[0, r][1][0][0]
        srates[p + "_y"] = data[0, r][1][0][0]
        srates[p + "_z"] = data[0, r][1][0][0]
        sample_size = len(data[0, r][2])
        for i in range(sample_size):
            params[p + "_x"].append(data[0, r][2][i][0])
            params[p + "_y"].append(data[0, r][2][i][1])
            params[p + "_z"].append(data[0, r][2][i][2])
    params["AUDIO"] = data[0, 0][2]
    srates["AUDIO"] = data[0, 0][1][0][0]
    return params, srates


def calc_gestures(mat_fname, trans_fname, filter_critical_points=False, m=0.05):
    gestures = {}

    # parse trans file
    t_starts, t_ends, phonemes, words, sentences = parse_transcription(trans_fname)
    phones = list(set(phonemes))

    # parse mat file
    params, srates = parse_mat(mat_fname)

    critical_points = set()
    if filter_critical_points:
        param_names = params.keys()
        param_names = [p[:-2] for p in param_names]
        param_names = list(set(param_names))
        for p in param_names:
            critical_points.update(cp.find_critical_points(p, params, m))


    for ph in phones:
        gestures[ph] = ges.Gesture(ph)

    # loop through phones
    for i in range(len(t_starts)):
        t_s = float(t_starts[i])
        t_e = float(t_ends[i])
        ph = phonemes[i]
        samples = {}
        # loop through ema from t_s up to t_e
        for p in params:
            rate = srates[p]
            i_start = int(t_s * rate)
            i_end = int(t_e * rate)

            # try to filter only samples considered as target
            length = i_end - i_start
            if filter_critical_points:
                samples[p] = [v for v, i in zip(params[p][i_start:i_start+length], range(i_start, i_end)) if
                              not math.isnan(v) and i in critical_points]
            else:
                samples[p] = [v for v in params[p][i_start:i_start + length] if not math.isnan(v)]
        gestures[ph].add_samples(samples)
    return gestures


def normalize_gestures(gestures):
    g = gestures.itervalues().next()
    params = g.phonemes[0].params.keys()
    p_max = {p: max(g.phonemes[0].params[p]) for p in params}
    p_min = {p: min(g.phonemes[0].params[p]) for p in params}
    for p in params:
        for g in gestures:
            for phoneme in gestures[g].phonemes:
                if len(phoneme.params[p]) > 0:
                    p_max[p] = max(p_max[p], max(phoneme.params[p]))
                    p_min[p] = min(p_min[p], min(phoneme.params[p]))

    norm_gestures = gestures.copy()
    for g in norm_gestures:
        for p in params:
            for i in range(len(norm_gestures[g].phonemes)):
                norm_gestures[g].phonemes[i].params[p] = [(v - p_min[p])/(p_max[p]-p_min[p]) for v in
                                                          norm_gestures[g].phonemes[i].params[p]]
    return norm_gestures, p_max, p_min


def parse_corpus(dir):
    phonemes_dict = {}
    gestures = {}
    fnames = list_TIMIT_dir(dir)

    for trans_fname, mat_fname in fnames:
        # parse transcription
        t_starts, t_ends, phonemes, words, sentences = parse_transcription(trans_fname)

        # parse mat file
        params, srates = parse_mat(mat_fname)

        # loop trough phonemes
        for t_start, t_end, phoneme_name in zip(t_starts, t_ends, phonemes):
            phoneme_params = {}
            for p in params:
                sampling_rate = srates[p]
                sample_start = int(t_start * sampling_rate)
                sample_start = max(0, sample_start-int(sampling_rate*0.01))
                sample_end = int(t_end * sampling_rate)
                sample_end = min(int(t_ends[-1]*sampling_rate)-1, sample_end + int(sampling_rate*0.01))
                trajectory = params[p][sample_start:sample_end]
                phoneme_params[p] = trajectory
            phoneme = ges.Phoneme(phoneme_name, phoneme_params, (t_start, t_end), trans_fname)

            if phoneme_name not in gestures:
                gestures[phoneme_name] = ges.Gesture(phoneme_name)
            gestures[phoneme_name].add(phoneme)
    return gestures


def scale_phonemes(num_scale, gestures):
    scaled_gestures = {}
    for g in gestures:
        scaled_gestures[g] = ges.Gesture(g)
        phonemes = gestures[g].phonemes
        for phoneme in phonemes:
            params = phoneme.params
            params_scaled = {}
            num_before = len(params["TT_x"])
            if num_before < 4:
                continue
            t_before = np.linspace(0, num_before, num_before)
            t_before_scaled = [t * num_scale / num_before for t in t_before]
            for p in params:

                # interpolate
                k = min(3, len(t_before_scaled)-1)
                tck = interpolate.splrep(t_before_scaled, params[p], s=0, k=k)
                t_after = np.linspace(0, num_scale, num_scale)
                params_scaled[p] = interpolate.splev(t_after, tck, der=0)
            scaled_phoneme = ges.Phoneme(g, params_scaled, phoneme.time, phoneme.source)
            scaled_gestures[g].add(scaled_phoneme)
    return scaled_gestures


def calc_means_and_variance(root):
    phonemes_dict = {}
    gestures = {}
    fnames = list_TIMIT_dir(dir)

    for trans_fname, mat_fname in fnames:
        # parse transcription
        t_starts, t_ends, phonemes, words, sentences = parse_transcription(trans_fname)

        # parse mat file
        params, srates = parse_mat(mat_fname)

        critical_points = []

        for p in params:
            trajectory = params[p]
            derivative = [params[p][i] - params[p][i-1] for i in range(1, len(params[p]))]
            for i in range(1, len(derivative)):
                if derivative[i]*derivative[i-1] <= 0:
                    critical_points.append(i)



        # loop trough phonemes
        for t_start, t_end, phoneme_name in zip(t_starts, t_ends, phonemes):
            phoneme_params = {}
            for p in params:
                sampling_rate = srates[p]
                sample_start = int(t_start * sampling_rate)
                sample_end = int(t_end * sampling_rate)
                trajectory = params[p][sample_start:sample_end]
                phoneme_params[p] = trajectory
            phoneme = ges.Phoneme(phoneme_name, phoneme_params, (t_start, t_end), trans_fname)

            if phoneme_name not in gestures:
                gestures[phoneme_name] = ges.Gesture(phoneme_name)
            gestures[phoneme_name].add(phoneme)
    return gestures

def test_parse_corpus():
    speaker = "F1"
    root_dir = "../USC-TIMIT/EMA/Data/" + speaker
    speaker += "ext"
    gestures = parse_corpus(root_dir)
    print "Gestures calculated"
    save_obj(gestures, speaker)
    print "Gestures saved"

    # gestures = load_obj("gestures")
    scaled_gestures = scale_phonemes(100, gestures)
    print "Gestures scaled"
    save_obj(scaled_gestures, speaker+"_s")
    print "Scaled gestures saved"

    norm_gestures,p_max,p_min = normalize_gestures(gestures)
    print "Gestures normed"
    save_obj(norm_gestures, speaker + "_n")
    save_obj(p_max, speaker + "_p_max")
    save_obj(p_min, speaker + "_p_min")
    print "Normed gestures saved"

    norm_gestures = None
    gestures = None

    norm_gestures_s, p_max_s, p_min_s = normalize_gestures(scaled_gestures)
    print "Gestures scaled and normed"
    scaled_gestures = None
    save_obj(norm_gestures_s, speaker + "_s_n")
    save_obj(p_max_s, speaker + "_p_max_s")
    save_obj(p_min_s, speaker + "_p_min_s")
    print "Scaled and normed gestures saved"

    print "Finished"
    return


def test():

    gestures = {}

    for fname in os.listdir(trans_dir):
        fname = os.path.splitext(fname)[0]
        print "Analyze ", fname
        t_fname = os.path.join(trans_dir, fname + ".trans")
        mat_fname = os.path.join(mat_dir, fname + ".mat")
        gest = calc_gestures(mat_fname, t_fname)
        for g in gest:
            if g not in gestures:
                gestures[g] = ges.Gesture(g)
            gestures[g].extend(gest[g])

    for g_name, g in gestures.items():
        print g_name
        print len(g.params["LL_x"])
        g_m = g.get_mean()
        print g_m
        g_v = g.get_variance()
        print g_v

    norm_gest, _, _ = normalize_gestures(gestures)

    for g_name, g in norm_gest.items():
        print g_name
        print len(g.params["LL_x"])
        g_m = g.get_mean()
        print g_m
        g_v = g.get_variance()
        print g_v
    return


def find_weights():
    import matplotlib.pyplot as plt


    root_dir = "../USC-TIMIT/EMA/Data/M1"
    index = 10

    t_names, m_names = zip(*list_TIMIT_dir(root_dir))
    trans_fname = t_names[index]
    mat_fname = m_names[index]

    gestures = {}
    means = {}  # key : param_name, value: dict(ges, val)
    variances = {}  # key : param_name, value: dict(ges, val)

    articulators = ["LL", "UL", "TT", "TB", "TD", "JAW"]
    domains = ["_x", "_y"]
    param_names = [a + d for a in articulators for d in domains]

    for i in range(len(t_names)):
        t_fname = t_names[i]
        mat_fname = m_names[i]
        gest = calc_gestures(mat_fname, t_fname, filter_critical_points=False, m=0.05)
        for g in gest:
            if g not in gestures:
                gestures[g] = ges.Gesture(g)
            gestures[g].extend(gest[g])
    print "gestures calculation finished"

    gestures_norm, p_max, p_min = normalize_gestures(gestures)

    # for p in param_names:
    #     means[p] = {}
    #     variances[p] = {}

    for g in gestures_norm:
        variances[g] = {}
        means[g] = {}
        g_m = gestures_norm[g].get_mean()
        g_v = gestures_norm[g].get_variance()
        for p in param_names:
            means[g][p] = g_m[p] * (p_max[p] - p_min[p]) + p_min[p]
            variances[g][p] = g_v[p]


    print "Means and variances calculated succesfully"
    for g in gestures_norm:
        print "{}    /////////////////////////////////".format(g)
        for k,v in variances[g].items():
            print "{} : {:.2f}".format(k, math.exp(-50.0*v))


def parse_audio(root_dir, spekar_name):

    fnames = list_TIMIT_dir(root_dir)

    phonemes_audio = {}

    for trans_fname, mat_fname in fnames:
        # parse transcription
        t_starts, t_ends, phonemes, words, sentences = parse_transcription(trans_fname)

        # parse mat file
        params, srates = parse_mat(mat_fname)

        audio = params["AUDIO"]
        srate = srates["AUDIO"]

        for i in range(len(t_starts)):
            idx_begin = int(t_starts[i]*srate)
            idx_end = int(t_ends[i]*srate)
            phoneme = phonemes[i]
            if phoneme not in phonemes_audio:
                phonemes_audio[phoneme] = []
            phonemes_audio[phoneme].append(audio[idx_begin:idx_end])

    save_obj(phonemes_audio, "{0}_phonemes".format(spekar_name))


def count_words(directory):
    phonemes_dict = {}
    gestures = {}
    fnames = list_TIMIT_dir(directory)
    words_dict = {}
    for trans_fname, mat_fname, _ in fnames:
        # parse transcription
        t_starts, t_ends, phonemes, words, sentences = parse_transcription(trans_fname)
        for i, w in enumerate(words):
            if words[i-1] != w:
                if w not in words_dict:
                    words_dict[w] = 0
                words_dict[w] += 1
    sorted_words = sorted(words_dict, key=words_dict.get, reverse=True)
    for w in sorted_words:
        print w, words_dict[w]

directory = r"C:\Study\DB\USC-TIMIT\USC-TIMIT\EMA\Data\M1"
# print directory
count_words(directory=directory)


def extract_words(directory, words, destination_folder):
    print "extracting words:" , words
    fnames = list_TIMIT_dir(directory)
    words_counter = {}
    for w in words:
        words_counter[w] = 0
    t_starts_dict = {}
    for trans_fname, mat_fname, wav_fname in fnames:
        # parse transcription
        t_start = 0.
        t_end = 0.
        t_starts, t_ends, phonemes, words_list, sentences = parse_transcription(trans_fname)
        for i, w in enumerate(words_list):
            if words_list[i - 1] != w and w in words:
                t_start = t_starts[i]
            if words_list[i - 1] != w and words_list[i - 1] in words:
                t_end = t_ends[i]
                fs, signal = wav.read(os.path.join(wav_fname))
                wav.write("{}/{}_{}.wav".format(destination_folder,
                                                words_list[i - 1],
                                                words_counter[words_list[i - 1]]),
                          fs,
                          signal[int(fs * t_start): int(fs * t_end)])
                words_counter[words_list[i - 1]] += 1
                t_start = 0.
                t_end = 0.

words = ["box", "bob", "be", "boy", "call", "gas", "take", "love", "book", "biology", "noise", "lost", "candy", "five", "lake", "lily"]
destination_folder = r"C:\Study\AAM\Data\timit_words"
directory = r"C:\Study\DB\USC-TIMIT\USC-TIMIT\EMA\Data\M1"
extract_words(directory, words, destination_folder)


# parse_audio("../USC-TIMIT/EMA/Data/F1", "F1")

import scipy.io.wavfile as wav
# from scipy.io import wavfile
# import random
# import numpy as np
#
# audio = load_obj("F1_phonemes")
# wav = np.array([item for sublist in audio["ih"] for item in sublist[len(sublist)/4 : len(sublist)*3/4] ])
# wav_int = np.int16(wav * (2 ** 15 - 1))
#
# wavfile.write('f1_ih.wav', 22050, wav_int)


# find_weights()

# test_parse_corpus()