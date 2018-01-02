import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf
import time
import random;
import numpy;



data, outputs = DataParser.ReadFile("sol100000.txt", 600, True);
#d = list(zip(data, outputs));
#random.shuffle(d);
#data, outputs = zip(*d);
a = Autoencoder.Autoencoder(217, [144, 96, 64, 42], Autoencoder.Autoencoder.crossEntropyLoss);
a.pretrain(0.01, 0, 5000, data[0:10001], 0, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
a.pretrain(0.003, 1, 10000, data[0:10001], 0, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
a.pretrain(0.001, 2, 10000, data[0:10001], 0, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
a.pretrain(0.001, 3, 5000, data[0:10001], 0, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
c = Classifier.Classifier(a, 14);
c.train(data[0:10001], outputs[0:10001], 0.01, 5000, "./summaries/finetuning");
print(c.test(data[0:10001], outputs[0:10001]));
print(c.test(data[10001: 12000], outputs[10001:12000]));
#c.save_model("smalldata_impressive results")

