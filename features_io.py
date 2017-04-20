import pprint

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, maxabs_scale

from features import mfcc_features, logfbank_features
from utils import get_fname_label_pairs

def generate_feature_mat(folder="a", train=True):
    """
    Generate feature matrix
    """
    training_df = get_fname_label_pairs(folder="a", train=train)

    features_df = pd.DataFrame()
    for i in range(0, len(training_df)):
        wav_file = "training/training-{}/{}.wav".format(folder, 
                training_df.iloc[i]["filename"])
        mfcc_feat = mfcc_features(wav_file)
        # normalization
        # don't do normalization. It reduces accuracy
        # mfcc_feat /= np.max(np.abs(mfcc_feat), axis=0)
        logfbank_feat = logfbank_features(wav_file)
        # normalization
        # don't do normalization. It reduces accuracy
        # logfbank_feat /= np.max(np.abs(logfbank_feat), axis=0)
        comb_feat = np.append(mfcc_feat, logfbank_feat)
        features_df = features_df.append([comb_feat], ignore_index=True)
    
    #pprint.pprint(features_df.shape)
    #pprint.pprint(features_df.head())

    # labels
    labels = pd.DataFrame()
    for i in range(0, len(training_df)):
        labels = labels.append([training_df.iloc[i]["abnormal"]], ignore_index=True)
    #pprint.pprint(labels)

    return features_df, labels

