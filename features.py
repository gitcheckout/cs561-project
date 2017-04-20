import pprint

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from python_speech_features import mfcc
from python_speech_features.base import logfbank, ssc
from scipy.io import wavfile
from scipy.signal import periodogram


def mfcc_features(fname):
    """
    Compute MFCC features
    """
    (rate, signal) = wavfile.read(fname)
    mfcc_feat = mfcc(signal, rate, numcep=13, appendEnergy=True)
    # make mean of all rows
    features = mfcc_feat.mean(axis=0)
    return features
    

def logfbank_features(fname):
    """
    Compute log Mel-filterbank energy features
    """
    (rate, signal) = wavfile.read(fname)
    fbank_beat = logfbank(signal, rate)
    # take mean of all rows
    features = fbank_beat.mean(axis=0)
    return features


def chroma_features(fname):
    """
    Compute chrome features
    """
    # (rate, signal) = wav.read(fname)
    signal, rate = librosa.load(fname)
    y_harmonic, y_percussive = librosa.effects.hpss(signal)
    # this is very slow
    # run it on small data
    chroma_feat = librosa.feature.chroma_stft(y_harmonic, rate)
    chroma_feat = pd.DataFrame(chroma_feat)

    chroma_feat_mss = pd.DataFrame()
    chroma_feat_mss['chroma_mean'] = chroma_feat.mean(axis=1)
    chroma_feat_mss['chroma_std'] = chroma_feat.std(axis=1)
    chroma_feat_mss['chroma_sum'] = chroma_feat.sum(axis=1)
    return chroma_feat_mss.values.flatten()


def spectral_features(fname):
    signal, rate = librosa.load(fname)

    # spectral centroid
    spec_centroid = librosa.feature.spectral.spectral_centroid(signal, rate)
    print(type(spec_centroid))
    print(spec_centroid.shape)


def pow_spec_density(fname):
    """
    Estimate power spectral density using a periodogram
    """
    rate, signal = wavfile.read(fname)
    (sample_freq, psd) = periodogram(signal, rate)
    print(type(psd))
    pprint.pprint(psd)
    pprint.pprint(psd.shape)


def ssc_features(fname):
    """
    Compute Spectral Subband Centroid features 
    """
    rate, signal = wavfile.read(fname)
    ssc_feat = ssc(signal, rate)
    # print(ssc_feat.shape)
    return ssc_feat.mean(axis=0)


def zcr_features(fname):
    """
    Compute zero crossing rate"
    """
    rate, signal = wavfile.read(fname)
    zcr_feat = librosa.zero_crossings(signal, pad=False)
    return np.atleast_1d(np.sum(zcr_feat))
