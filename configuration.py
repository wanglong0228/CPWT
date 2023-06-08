import os
import datetime
import numpy as np
import tensorflow as tf

## These lines ensure any existing tf sessions are closed.
try:
    tf.Session().close()
except:
    pass

TEST_PDB_CODE_DB5 = ['3R9A', '4GAM', '3AAA', '4H03', '1EXB', '2GAF', '2GTP', '3RVW', '3SZK', '4IZ7', '4GXU',
                    '3BX7', '2YVJ','3V6Z', '1M27', '4FQI', '4G6J', '3BIW', '3PC8', '3HI6', '2X9A', '3HMX',
                    '2W9E', '4G6M', '3LVK', '1JTD','3H2V', '4DN4', 'BP57', '3L5W', '3A4S', 'CP57', '3DAW',
                    '3VLB', '3K75', '2VXT', '3G6D', '3EO1', '4JCV','4HX3', '3F1P', '3AAD', '3EOA', '3MXW',
                    '3L89', '4M76', 'BAAD', '4FZA', '4LW4', '1RKE', '3FN1', '3S9D','3H11', '2A1A', '3P57']

# data_directory = '/opt/data/share/120106022644/protein/datasets/pipgcn/DB5.5/'
data_directory = '/opt/data/share/120106022644/protein/datasets/pipgcn/DB5/'
# data_directory = '/opt/data/share/120106022644/protein/datasets/pipgcn/DB4/'
# data_directory = '/opt/data/share/120106022644/protein/datasets/pipgcn/DB3/'
output_directory = './output/'

## Random Seeds
# each random seed represents an experimental replication.
# You can add or remove list elements to change the number
# of replications for an experiment.
# seeds = [
#     {"tf_seed": 649737, "np_seed": 29820},
#     {"tf_seed": 395408, "np_seed": 185228},
#     {"tf_seed": 252356, "np_seed": 703889},
#     {"tf_seed": 343053, "np_seed": 999360},
#     {"tf_seed": 743746, "np_seed": 67440},
#     {"tf_seed": 364047, "np_seed": 4047},
#     {"tf_seed": 297847, "np_seed": 645393},
#     {"tf_seed": 764859, "np_seed": 786543},
#     # {"tf_seed": 175343, "np_seed": 378945},
#     # {"tf_seed": 856516, "np_seed": 597688},
#     # {"tf_seed": 474313, "np_seed": 349903},
#     {"tf_seed": 838382, "np_seed": 897904},
#     {"tf_seed": 202003, "np_seed": 656146},
# ]


seeds = [ 
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 395408, "np_seed": 185228},
]
GCN_layers = [
    {1: [[None, 256]]},
    # {2: [[None, 256], [256, 512]]},
    # {3: [[None, 256], [256, 256], [256, 512]]},
    # {4: [[None, 256], [256, 256], [256, 512], [512, 512]]}
]
# A slightly fancy printing method
def printt(msg):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}| {}".format(time_str, msg))
