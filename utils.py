import os
import random

import pandas as pd


def split_train_test(folder='a', ratio=0.2):
    # records = 'training/training-' + folder + '/RECORDS'
    records_normal = 'training/training-' + folder + '/RECORDS-normal'
    records_abnormal = 'training/training-' + folder + '/RECORDS-abnormal'

    read_normal = open(records_normal, 'r')
    normal_files = read_normal.readlines()
    normal_files = [fname.strip() for fname in normal_files]
    random.shuffle(normal_files)

    n = int(len(normal_files)*ratio)
    
    train_normal_files = normal_files[n:]
    train_normal_fname = "train_normal_" + folder + ".txt"
    train_normal_fconn = open(train_normal_fname, 'w')
    for fname in train_normal_files:
        train_normal_fconn.write("{}\n".format(fname))
    train_normal_fconn.close()

    test_normal_files = normal_files[:n]
    test_normal_fname = "test_normal_" + folder + ".txt"
    test_normal_fconn = open(test_normal_fname, 'w')
    for fname in test_normal_files:
        test_normal_fconn.write("{}\n".format(fname))
    test_normal_fconn.close()

    read_abnormal = open(records_abnormal, 'r')
    abnormal_files = read_abnormal.readlines()
    abnormal_files = [fname.strip() for fname in abnormal_files]
    random.shuffle(abnormal_files)

    n = int(len(abnormal_files)*ratio)
    
    train_abnormal_files = abnormal_files[n:]
    train_abnormal_fname = "train_abnormal_" + folder + ".txt"
    train_abnormal_fconn = open(train_abnormal_fname, 'w')
    for fname in train_abnormal_files:
        train_abnormal_fconn.write("{}\n".format(fname))
    train_abnormal_fconn.close()

    test_abnormal_files = abnormal_files[:n]
    test_abnormal_fname = "test_abnormal_" + folder + ".txt"
    test_abnormal_fconn = open(test_abnormal_fname, 'w')
    for fname in test_abnormal_files:
        test_abnormal_fconn.write("{}\n".format(fname))
    test_abnormal_fconn.close()


def get_fname_label_pairs(folder='a', train=True):
    dtype = "train" if train else "test"
    
    normal_fname = dtype + "_normal_" + folder + ".txt"
    abnormal_fname = dtype + "_abnormal_" + folder + ".txt"
   
    if not os.path.exists(normal_fname):
        print("File {} does not exists.".format(normal_fname))
    normal_fconn = open(normal_fname, 'r')
    normal_files = normal_fconn.readlines()
    normal_files = [fname.strip() for fname in normal_files]

    if not os.path.exists(abnormal_fname):
        print("File {} does not exists.".format(abnormal_fname))
    abnormal_fconn = open(abnormal_fname, 'r')
    abnormal_files = abnormal_fconn.readlines()
    abnormal_files = [fname.strip() for fname in abnormal_files]

    exp_data = []

    for fname in normal_files:
        exp_data.append([fname, 0])
    for fname in abnormal_files:
        exp_data.append([fname, 1])

    exp_data = pd.DataFrame(exp_data)
    exp_data.columns = ['filename', 'abnormal']
    return exp_data
