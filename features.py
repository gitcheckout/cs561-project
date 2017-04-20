
import pprint

import librosa
from python_speech_features import mfcc
from python_speech_features.base import logfbank
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
    #(rate, signal) = wav.read(fname)
    signal, rate = librosa.load(fname)
    y_harmonic, y_percussive = librosa.effects.hpss(signal)
    chroma_feat = librosa.feature.chroma_cqt(signal, rate)

    #print(type(chroma_feat))
    #print(len(chroma_feat))
    #print(chroma_feat.shape)
    #pprint.pprint(chroma_feat)
    return


def spectral_features(fname):
    signal, rate = librosa.load(fname)

    # spectral centroid
    spec_centroid = librosa.feature.spectral.spectral_centroid(signal, rate)
    print(type(spec_centroid))
    print((spec_centroid.shape))

def pow_spec_density(fname):
    """
    Estimate power spectral density using a periodogram
    """
    rate, signal = wavfile.read(fname)
    (sample_freq, psd) = periodogram(signal, rate)
    print(type(psd))
    pprint.pprint(psd)
    pprint.pprint(psd.shape)

