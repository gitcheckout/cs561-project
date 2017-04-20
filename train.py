import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from features_io import generate_feature_mat


def train_and_classify(folder='a'):
    # training data
    training_data, training_labels = generate_feature_mat(folder)
    #assert len(training_data) == len(training_labels)

    testing_data, testing_labels = generate_feature_mat(folder, train=False)
    #assert len(testing_data) == len(testing_labels)

    print("Features extracted")
    clf = svm.SVC()
    clf.fit(training_data, training_labels.values.ravel())
    print("Training completed")
    predictions = clf.predict(testing_data)

    score1 = accuracy_score(testing_labels, predictions)
    score2 = f1_score(testing_labels, predictions)
    score3 = matthews_corrcoef(testing_labels, predictions)
    print(score1, score2, score3)
