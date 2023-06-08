import os
import sys
import time
import pickle
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
from configuration import data_directory, output_directory, seeds, printt, GCN_layers
from train_test import TrainTest
from model import PW_classifier, Weight_Cross_Entropy
from results_processor import ResultsProcessor
import pdb
tf.get_logger().setLevel('ERROR')


# setup output directory
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
results_processor = ResultsProcessor()

# create results log
tmp = sys.argv[1]
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
results_log = os.path.join(output_directory, "{:}_{:}_results.csv".format(now, tmp))

# load dataset
printt("Load train data")
train_data_file = os.path.join(data_directory, "train.pkl")
_, train_data = pickle.load(open(train_data_file, 'rb'))
printt("Load test data")
test_data_file = os.path.join(data_directory, "test.pkl")
_, test_data = pickle.load(open(test_data_file, 'rb'))

data = {"train": train_data, "test": test_data}
# pdb.set_trace() 
# perform experiment for each random seed  
for i, seed_pair in enumerate(seeds):
    for j, gcn_layer in enumerate(GCN_layers):
        printt("rep{:}/layer_{:}".format(i, j + 1))
        # set tensorflow and numpy seed
        tf.random.set_seed(seed_pair['tf_seed'])
        np.random.seed(int(seed_pair['np_seed']))
        
        printt("build model")
        # configure
        in_dims = 70 
        learning_rate = 0.1
        
        # piece_wise_constant_decay = PiecewiseConstantDecay()
        pn_ratio = 0.1 #ratio between
        
        
        
        
        
        # neg and pos 设置正负样本10：1
        # build model
        model = PW_classifier(in_dims=in_dims, gcn_layer_num=j + 1, gcn_config=gcn_layer[j + 1])
        cerition = Weight_Cross_Entropy(pn_ratio=pn_ratio)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        # train and test the model
        headers, results = TrainTest(results_log, j + 1, results_processor).fit_model(data, model, cerition, optimizer)
    
