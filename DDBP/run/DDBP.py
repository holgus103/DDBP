
import tensorflow as tf
import time
import random;
import numpy;
import sys;
import pprint;


# configure here
TEST_TRUMP = True
TRAIN_TRUMP = False
TEST_NO_TRUMP = False
TRAIN_NO_TRUMP = True
BATCHES = 1
PARTITION = 0.66
SET_SIZE = 1000
EXPERIMENT = "experiment_name"

def suit_count_for_params(no_trump, trump):
    acc = 0;
    if(no_trump):
        acc = acc + 1;
    if(trump):
        acc = acc + 4;
    return acc;

# main experiment code
sys.path.append("./../");

import models;
import data_parser as dp;

experiment_name = EXPERIMENT;
path = "./summaries/{0}/".format(experiment_name);

dp.initialize_random(experiment_name);

# import data
(data, outputs, test_data, test_outputs) = dp.read_file("./../data/library", SET_SIZE, True, TRAIN_NO_TRUMP, TRAIN_TRUMP, TEST_NO_TRUMP, TEST_TRUMP, PARTITION);

d_train = dp.get_distribution(data, outputs);
d_test = dp.get_distribution(test_data, test_outputs);
dp.save_distribution(path, d_train, d_test);
optimizer = tf.train.RMSPropOptimizer 
print(len(data));
print(len(outputs));
print(len(test_data));
print(len(test_outputs))
# calculate test set length
l = len(data);
batch_count = BATCHES;
data_batches = [];
outputs_batches = [];

# separate data into batches
batch_size = int(l / batch_count);
for i in range(0, batch_count-1):
    data_batches.append(data[i * batch_size : (i + 1) * batch_size]);
    outputs_batches.append(outputs[i * batch_size : (i + 1) * batch_size]);

data_batches.append(data[(batch_count - 1) * batch_size : l]);
outputs_batches.append(outputs[(batch_count - 1) * batch_size : l]);
print(len(data_batches[0]))
#print(len(data_batches[1]))
# create autoencoder
# a = models.Autoencoder(217, [197, 179, 162, 147], models.Model.cross_entropy_loss);
a = models.Autoencoder(217, [174], models.Model.cross_entropy_loss);


# pretrain each layer
a.pretrain(0.001, 0, 7000, data_batches, 0, 0.01, path + "{0}" , optimizer, 0.2, 15);

# create classifier
c = models.Classifier(a, 14);
# train whole network
c.train(data_batches, outputs_batches, 0.0001, 15000, 0.0001, path +"/finetuning", data, outputs, test_data, test_outputs, suit_count_for_params(TRAIN_NO_TRUMP, TRAIN_TRUMP), suit_count_for_params(TEST_NO_TRUMP, TEST_TRUMP), models.Model.mse_loss, 25, experiment_name);

# evaluate results
print(c.test(data, outputs));
print(c.test(test_data, test_outputs));
print(c.suit_based_accurancy(data, outputs, suit_count_for_params(TRAIN_NO_TRUMP, TRAIN_TRUMP)));
print(c.suit_based_accurancy(test_data, test_outputs, suit_count_for_params(TEST_NO_TRUMP, TRAIN_NO_TRUMP)));
c.save_model(experiment_name);


