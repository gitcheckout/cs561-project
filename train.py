from sklearn import svm
from sklearn.metrics import accuracy_score

from features_io import generate_mfcc_features

def train():
    # training data
    training_data, training_labels = generate_mfcc_features()
    assert len(training_data) == len(training_labels)
    #print(type(get_fname_label_pairs()))

    #training_data = training_tuple[0]
    #training_labels = training_tuple[1]
    
    #return

    #for i in range(0, len(training_labels)):
     #   print(training_labels[0][0])
    #return
    testing_data, testing_labels = generate_mfcc_features(train=False)
    assert len(testing_data) == len(testing_labels)
    #testing_data = testing_tuple[0]
    #testing_labels = testing_tuple[1]

    clf = svm.SVC()
    clf.fit(training_data, training_labels)
    predictions = clf.predict(testing_data)
    
    score = accuracy_score(testing_labels, predictions)
    print(score)

train()

