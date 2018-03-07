
import tensorflow as tf
import time
import random;
import numpy;
import sys;
import pprint;


sys.path.append("./../");

import models;
import data_parser as dp;

experiment_name = "shallow54_no_trump_100k";
path = "./summaries/{0}/".format(experiment_name);

dp.reset_random(experiment_name);

# import data
(data, outputs, test_data, test_outputs) = dp.read_file("./../data/library", 200000, True, True, False, True, True, 0.5);

d_train = dp.get_distribution(data, outputs);
d_test = dp.get_distribution(test_data, test_outputs);
dp.save_distribution(path, d_train, d_test);
#print(len(data_batches[1]))
# create autoencoder
# a = models.Autoencoder(217, [197, 179, 162, 147], models.Model.cross_entropy_loss);
a = models.Autoencoder(217, [54], models.Model.cross_entropy_loss);


# create classifier
c = models.Classifier(a, 14);
c.restore_model(experiment_name)
# evaluate results
print(c.test(data, outputs));
print(c.test(test_data, test_outputs));
print(c.suit_based_accurancy(data, outputs, 1));
print(c.suit_based_accurancy(test_data, test_outputs, 5));
#c.save_model(experiment_name);


