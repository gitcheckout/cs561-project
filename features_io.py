import pprint

import numpy as np
import pandas as pd

from features import mfcc_features
from utils import get_fname_label_pairs

def generate_mfcc_features(folder="a", train=True):
    training_df = get_fname_label_pairs(folder="a", train=train)

    features_df = pd.DataFrame()
    for i in range(0, len(training_df)):
        wav_file = "training/training-{}/{}.wav".format(folder, 
                training_df.iloc[i]["filename"])
        mfcc_feat = mfcc_features(wav_file)
        features_df = features_df.append([mfcc_feat], ignore_index=True)
    #pprint.pprint(features_df.shape)
    #pprint.pprint(features_df.head())

    # labels
    labels = pd.DataFrame()
    for i in range(0, len(training_df)):
        labels = labels.append([training_df.iloc[i]["abnormal"]], ignore_index=True)
    #pprint.pprint(labels)

    return features_df, labels

#a, b = generate_mfcc_features()
#pprint.pprint(b)

