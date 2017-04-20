import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

from features_io import generate_feature_mat

def train_and_classify():
    # training data
    training_data, training_labels = generate_feature_mat()
    assert len(training_data) == len(training_labels)

    testing_data, testing_labels = generate_feature_mat(train=False)
    assert len(testing_data) == len(testing_labels)

    clf = svm.SVC()
    clf.fit(training_data, training_labels)
    predictions = clf.predict(testing_data)
    
    score = accuracy_score(np.ravel(testing_labels), predictions)
    print(score)

