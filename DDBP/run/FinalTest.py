
import tensorflow as tf
import time
import random;
import numpy;
import sys;

sys.path.append("./../");

import models;
import data_parser as dp






model_names = ["colors_3", "no_trump", "50k"]
data, outputs = dp.read_file("./../data/sol100000.txt", 50000, True, True, True);

l = len(data);



a = models.Autoencoder(217, [174, 140, 112, 90], models.Model.cross_entropy_loss);
c = models.Classifier(a, 14);

for name in model_names:
    c.restore_model(name);
    print(name)
    print("Results:")
    print(c.test(data[0:test_end], outputs[0:test_end]));
    res = c.suit_based_accurancy(data[0:test_end], outputs[0:test_end], 5);
    print(res);