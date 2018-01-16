

import tensorflow as tf
import time
import random;
import numpy;
import Autoencoder;
import Classifier;
import DataParser;

def test(c, test_data, test_labels, suits):
    l = len(test_data);
    for i in range(0, suits):        
        input = [test_data[x] for x in range(0, l) if x % (suits * 4) in range(4 * i, 4*i + 4)];
        labels = [test_labels[x] for x in range(0, l) if x % (suits * 4) in range(4 * i, 4*i + 4)];
        print(c.test(input, labels));
    

data, outputs = DataParser.ReadFile("sol100000.txt", 50000, True);

l = len(data);
test_end = int(l * 0.66);

a = Autoencoder.Autoencoder(217, [174, 140, 112, 90], Autoencoder.Autoencoder.crossEntropyLoss);
c = Classifier.Classifier(a, 14);
c.restore_model("no_trump");

test(c, data[0:test_end], outputs[0:test_end], 5);
test(c, data[test_end:l], outputs[test_end:l], 5);



