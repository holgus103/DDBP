import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf
import time
import random;
import numpy;



data, outputs = DataParser.ReadFile("sol100000.txt", 50000, True, True, False);
#d = list(zip(data, outputs));
#random.shuffle(d);
#data, outputs = zip(*d);
l = len(data);
test_end = int(l * 0.66);
batch_count = 2;
data_batches = [];
outputs_batches = [];
# separate data into batches
batch_size = int(test_end / batch_count);
for i in range(0, batch_count-1):
    data_batches.append(data[i * batch_size : (i + 1) * batch_size]);
    outputs_batches.append(outputs[i * batch_size : (i + 1) * batch_size]);

data_batches.append(data[(batch_count - 1) * batch_size : test_end]);
outputs_batches.append(outputs[(batch_count - 1) * batch_size : test_end]);

a = Autoencoder.Autoencoder(217, [174, 140, 112, 90], Autoencoder.Autoencoder.crossEntropyLoss);
a.pretrain(0.01, 0, 5000, data_batches, 0, "./summaries/no_trump/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
a.pretrain(0.005, 1, 7500, data_batches, 0, "./summaries/no_trump/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
a.pretrain(0.001, 2, 10000, data_batches, 0, "./summaries/no_trump/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
a.pretrain(0.001, 3, 12500, data_batches, 0, "./summaries/no_trump/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
c = Classifier.Classifier(a, 14);
c.train(data_batches, outputs_batches, 0.001, 10000, "./summaries/no_trump/finetuning");
print(c.test(data[0:test_end], outputs[0:test_end]));
print(c.test(data[test_end: l], outputs[test_end:l]));
c.save_model("no_trump");


