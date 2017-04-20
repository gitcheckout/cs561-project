from train import train_and_classify
from utils import split_train_test

from features import zcr_features

GEN_SPLIT_CASE = False


def start_test():
    feat = zcr_features('training/training-a/a0001.wav')
    # print(feat)
    return


def start():
    """
    starting function
    :return: None
    """
    folder = 'a'
    if GEN_SPLIT_CASE:
        split_train_test(folder)
        print("Split completed")
    train_and_classify(folder, all_folders=True)
    return

if __name__ == "__main__":
    start()

