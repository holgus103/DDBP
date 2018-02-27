
import tensorflow as tf
import time
import random;
import numpy;
import sys;


sys.path.append("./../");

import models;
import data_parser as dp;

# import data
(data, outputs, test_data, test_outputs) = dp.read_file("./../data/sol100000.txt", 500, True, True, False);

# calculate test set length
l = len(data);
test_end = int(l * 0.66);
batch_count = 2;
data_batches = [];
outputs_batches = [];

# separate data into batches
batch_size = int(l / batch_count);
for i in range(0, batch_count-1):
    data_batches.append(data[i * batch_size : (i + 1) * batch_size]);
    outputs_batches.append(outputs[i * batch_size : (i + 1) * batch_size]);

data_batches.append(data[(batch_count - 1) * batch_size : test_end]);
outputs_batches.append(outputs[(batch_count - 1) * batch_size : test_end]);
# create autoencoder
a = models.Autoencoder(217, [174, 140, 112, 90], models.Model.cross_entropy_loss);

# pretrain each layer
a.pretrain(0.01, 0, 7000, data_batches, 0, 1, "./summaries/colors_4/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2, 5);
a.pretrain(0.005, 1, 8500, data_batches, 0, 1, "./summaries/colors_4/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2, 5);
a.pretrain(0.001, 2, 10000, data_batches, 0, 1, "./summaries/colors_4/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2, 5);
a.pretrain(0.001, 3, 12500, data_batches, 0, 1, "./summaries/colors_4/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2, 5);

# create classifier
c = models.Classifier(a, 14);

# train whole network
c.train(data_batches, outputs_batches, 0.001, 15000, 0.1, test_data, test_outputs "./summaries/colors_4/finetuning");

# evaluate results
print(c.test(data, outputs));
print(c.test(test_data, test_outputs));
print(c.suit_based_accurancy(data, outputs, 1));
print(c.suit_based_accurancy(test_data, test_outputs, 5));
c.save_model("colors_4");


