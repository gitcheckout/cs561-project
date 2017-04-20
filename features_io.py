import pprint

import numpy as np
import pandas as pd

from features import mfcc_features, logfbank_features, zcr_features, ssc_features
from utils import get_fname_label_pairs


def generate_feature_mat(folder="a", train=True):
    """
    Generate feature matrix
    """
    training_df = get_fname_label_pairs(folder=folder, train=train)

    features_df = pd.DataFrame()
    for i in range(0, len(training_df)):
        wav_file = "training/training-{}/{}.wav".format(
            folder, training_df.iloc[i]["filename"])
        mfcc_feat = mfcc_features(wav_file)
        # print("mfcc_feat done for {}".format(i))
        # normalization
        # do not do normalization. It reduces accuracy
        # mfcc_feat /= np.max(np.abs(mfcc_feat), axis=0)
        logfbank_feat = logfbank_features(wav_file)
        # print("logfbank_feat done for {}".format(i))
        # normalization
        # do not do normalization. It reduces accuracy
        # logfbank_feat /= np.max(np.abs(logfbank_feat), axis=0)
        ssc_feat = ssc_features(wav_file)
        zcr_feat = zcr_features(wav_file)
        # print("ssc_feat done for {}".format(i))

        comb_feat = np.append(mfcc_feat, logfbank_feat)
        comb_feat = np.append(comb_feat, ssc_feat)
        # comb_feat = np.append(comb_feat, zcr_feat)
        features_df = features_df.append([comb_feat], ignore_index=True)
    
    # pprint.pprint(features_df.shape)
    # pprint.pprint(features_df.head())

    # labels
    labels = pd.DataFrame()
    for i in range(0, len(training_df)):
        labels = labels.append([training_df.iloc[i]["abnormal"]], ignore_index=True)
    # pprint.pprint(labels)

    return features_df, labels

