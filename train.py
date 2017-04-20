import pprint
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from features_io import generate_feature_mat


def train_and_classify(folder='a', all_folders=False, folders=None):

    if not folders:
        folders = ['a', 'b', 'c', 'd', 'e', 'f']

    if not all_folders:
        training_data, training_labels = generate_feature_mat(folder)
    else:
        training_df_names = []
        training_label_names = []
        for folder in folders:
            folder_td, folder_tl = generate_feature_mat(folder)
            print("Type of tl is {}".format(type(folder_tl)))
            training_df_names.append(folder_td)
            training_label_names.append(folder_tl)
        
        training_labels = pd.concat(training_label_names)
        training_data = pd.concat(training_df_names)
        print("Shape of training data is: {}".format(training_data.shape))
        print("Shape of training labels is: {}".format(training_labels.shape))
        print("Training Labels are: ")


    if not all_folders:
        testing_data, testing_labels = generate_feature_mat(folder, train=False)
    else:
        testing_df_names = []
        testing_label_names = []
        for folder in folders:
            folder_td, folder_tl = generate_feature_mat(folder)
            testing_df_names.append(folder_td)
            testing_label_names.append(folder_tl)

        testing_labels = pd.concat(testing_label_names)
        testing_data = pd.concat(testing_df_names)
        print("Shape of testing data is: {}".format(testing_data.shape))

    print("Classifier running:")
    # SVM classifier
    clf = svm.SVC()
    print(training_labels.shape)
    clf.fit(training_data, training_labels.values.ravel())
    
    print("Predicting")
    predictions = clf.predict(testing_data)
    
    # scores
    score1 = accuracy_score(testing_labels, predictions)
    score2 = f1_score(np.atleast_1d(testing_labels), predictions)
    score3 = matthews_corrcoef(np.atleast_1d(testing_labels), predictions)
    score4 = physionet_scoring(testing_labels, predictions)
    print(score1, score2, score3, score4)


def physionet_scoring(testing_labels, predictions):

    testing_labels = testing_labels.values.tolist() 
    testing_labels = [item for sublist in testing_labels for item in sublist]
    predictions = predictions.tolist()
    nn1 = 0
    na1 = 0
    an1 = 0
    aa1 = 0
    print(testing_labels[1:10])
    print(predictions[1:10])
    for real, pred in zip(testing_labels, predictions):
        if real == 0 and pred == 0:
            nn1 += 1
        elif real == 1 and pred == 0:
            an1 += 1
        elif real == 0 and pred == 1:
            na1 += 1
        elif real == 1 and pred == 1:
            aa1 += 1
    
    se = float(aa1)/float(aa1+an1)
    sp = float(nn1)/float(na1+nn1)
    return (se+sp)/2

