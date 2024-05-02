import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

import numpy as np
from preprocess import ArtDataPreprocessor
from hyperparams import *

def train_test_code(create_new_data: bool) -> dict:
    "Takes in a boolean on whether new files are required, and returns the index_to_class label dictionary"
    data_dir = '../wikiart'
    preprocessor = ArtDataPreprocessor(data_dir, CLASSES)
    if create_new_data:
        # get preprocessed train and test data
        train_data, train_labels = preprocessor.get_train_data()
        test_data, test_labels = preprocessor.get_test_data()
        train_data = train_data.astype(np.float32)
        test_data = test_data.astype(np.float32)

        # save numpy array as csv file

        np.save('train_data.npy', train_data)
        np.save('test_data.npy', test_data)
        np.save('train_labels.npy', train_labels)
        np.save('test_labels.npy', test_labels)
    label_dict = preprocessor.idx_to_class
    return label_dict