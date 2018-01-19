
import tensorflow as tf
import time
import random;
import numpy;
import models;
import data_parser as dp






model_names = ["colors_2", "50k", "no_trump"]
data, outputs = dp.read_file("./data/sol100000.txt", 50000, True);

l = len(data);
test_end = int(0.66 * l);

batch_count = 4;
test_batches_count = 1;
data_batches = [];
outputs_batches = [];
test_batches = [];
outputs_test_batches = [];
# separate data into batches
# training set
dp.divide_into_batches(batch_count, data_batches, outputs_batches, data[0:test_end], outputs[0:test_end]);
# test set
dp.divide_into_batches(test_batches_count, test_batches, outputs_test_batches, data[test_end:l], outputs[test_end:l])


a = models.Autoencoder(217, [174, 140, 112, 90], models.Model.cross_entropy_loss);
c = models.Classifier(a, 14);

for name in model_names:
    c.restore_model(name);
    print(name)
    print("Training set:")
    res = c.suit_based_accurancy(data[0:test_end], outputs[0:test_end], 5);
    print(res);
    print("Test set")
    res = c.suit_based_accurancy(data[0:test_end], outputs[0:test_end], 5);
    print(res)
