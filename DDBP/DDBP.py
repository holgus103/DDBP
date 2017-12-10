import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf
import time




data, outputs = DataParser.ReadFile("sol100000.txt", 60);
a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.crossEntropyLoss);
a.pretrain([0.01, 0.01, 0.01, 0.01], [5000, 3000, 4000, 4000], data[0:1001], [0.0001, 0.01, 5, 10], "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer);
c = Classifier.Classifier(a, 14);
c.train(data[0:1001], outputs[0:1001], 0.01, 10000, "./summaries/finetuning");
c.test(data[1001: 1200], outputs[1001:1200]);

