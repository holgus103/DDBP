
import tensorflow as tf
import time
import random;
import numpy;
import Autoencoder;
import Classifier;
import DataParser;


data, outputs = DataParser.ReadFile("sol100000.txt", 50000, True);

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
c = Classifier.Classifier(a, 14);
c.restore_model("50k");
print(c.test(data[0:test_end], outputs[0:test_end]));
print(c.test(data[test_end: l], outputs[test_end:l]));
c.train(data_batches, outputs_batches, 0.001, 10000, "./summaries/finetuning", Autoencoder.Autoencoder.crossEntropyLoss);
print(c.test(data[0:test_end], outputs[0:test_end]));
print(c.test(data[test_end: l], outputs[test_end:l]));
c.save_model("50kf")

