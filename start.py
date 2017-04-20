import librosa
import matplotlib.pyplot as plt
from sys import argv
from train import train_and_classify
from utils import split_train_test

from features import chroma_features, zcr_features

GEN_SPLIT_CASE = True


def start_test():
    feat = zcr_features('training/training-a/a0001.wav')
    # print(feat)
    return


def start():
    folder=argv[1]
    if GEN_SPLIT_CASE:
        split_train_test(folder)
        print("Split completed")
    train_and_classify(folder)
    return

if __name__ == "__main__":
    start()

