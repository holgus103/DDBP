
import tensorflow as tf
import time
import random;
import numpy;
import sys;
import pprint;


sys.path.append("./../");

import models;
import data_parser as dp;

experiment_name = "shallow174_no_trump_100k";
path = "./summaries/{0}/".format(experiment_name);

dp.initialize_random(experiment_name);

# import data
(data, outputs, test_data, test_outputs) = dp.read_file("./../data/library", 200000, True, True, False, True, True, 0.5);

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
batch_count = 4;
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
c.train(data_batches, outputs_batches, 0.0001, 15000, 0.0001, path +"/finetuning", data, outputs, test_data, test_outputs, 1, 5, models.Model.mse_loss, 25, experiment_name);

# evaluate results
print(c.test(data, outputs));
print(c.test(test_data, test_outputs));
print(c.suit_based_accurancy(data, outputs, 1));
print(c.suit_based_accurancy(test_data, test_outputs, 5));
c.save_model(experiment_name);


