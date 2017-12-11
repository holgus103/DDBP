import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf
import time
import random;




data, outputs = DataParser.ReadFile("sol100000.txt", 60);
d = list(zip(data, outputs));
random.shuffle(d);
data, outputs = zip(*d);
a = Autoencoder.Autoencoder(217, [104, 52], Autoencoder.Autoencoder.crossEntropyLoss);
a.pretrain([0.01, 0.01, 0.01], [5000, 3000], data[0:1001], [0.0001, 0.01, 5, 10], "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer);
c = Classifier.Classifier(a, 14);
c.train(data[0:1001], outputs[0:1001], 0.01, 10000, "./summaries/finetuning");
print(c.test(data[1001: 1200], outputs[1001:1200]));
#c.save_model("smalldata_impressive results")

