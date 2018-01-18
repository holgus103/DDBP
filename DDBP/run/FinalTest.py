
import tensorflow as tf
import time
import random;
import numpy;
import models;
import data_parser as dp


data, outputs = dp.ReadFile("sol100000.txt", 100000, True);

l = len(data);

batch_count = 4;
data_batches = [];
outputs_batches = [];
# separate data into batches

batch_size = int(l / batch_count);
for i in range(0, batch_count-1):
    data_batches.append(data[i * batch_size : (i + 1) * batch_size]);
    outputs_batches.append(outputs[i * batch_size : (i + 1) * batch_size]);

data_batches.append(data[(batch_count - 1) * batch_size : l]);
outputs_batches.append(outputs[(batch_count - 1) * batch_size : l]);

a = models.Autoencoder(217, [174, 140, 112, 90], models.Model.cross_entropy_loss);
c = models.Classifier(a, 14);
c.restore_model("50k");
v1 = c.test(data_batches[0], outputs_batches[0]);
print(v1);
v2 = c.test(data_batches[1], outputs_batches[1]);
print(v2);
v3 = c.test(data_batches[2], outputs_batches[2]);
print(v3);
v4 = c.test(data_batches[3], outputs_batches[3]);
print(v1);

print("final")
print([sum(x) / 4 for x in zip(v1, v2, v3, v4)]);