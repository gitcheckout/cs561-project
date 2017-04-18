
import pprint

import librosa
from python_speech_features import mfcc
from python_speech_features.base import logfbank
import scipy.io.wavfile as wav


def mfcc_features(fname):
    (rate, signal) = wav.read(fname)
    mfcc_feat = mfcc(signal, rate, numcep=13, appendEnergy=True)
    return mfcc_feat.mean(axis=0)
    

def chroma_features(fname):
    #(rate, signal) = wav.read(fname)
    signal, rate = librosa.load(fname)
    y_harmonic, y_percussive = librosa.effects.hpss(signal)
    chroma_feat = librosa.feature.chroma_cqt(signal, rate)

    print(type(chroma_feat))
    print(len(chroma_feat))
    print(chroma_feat.shape)
    pprint.pprint(chroma_feat)



def spectral_features(fname):
    signal, rate = librosa.load(fname)

    # spectral centroid
    spec_centroid = librosa.feature.spectral.spectral_centroid(signal, rate)
    print(type(spec_centroid))
    print((spec_centroid.shape))

#spectral_features("training/training-a/a0003.wav")

