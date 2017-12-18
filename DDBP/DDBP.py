import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf
import time
import random;
import numpy;



data, outputs = DataParser.ReadFile("sol100000.txt", 600);
#d = list(zip(data, outputs));
#random.shuffle(d);
#data, outputs = zip(*d);
a = Autoencoder.Autoencoder(217, [104, 52], Autoencoder.Autoencoder.crossEntropyLoss);
a.pretrain(0.01, 0, 5000, data[0:10001], 5, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
a.pretrain(0.01, 1, 5000, data[0:10001], 5, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
c = Classifier.Classifier(a, 14);
c.train(data[0:10001], outputs[0:10001], 0.01, 5, "./summaries/finetuning");
print(c.test(data[10001: 12000], outputs[10001:12000]));
#c.save_model("smalldata_impressive results")

