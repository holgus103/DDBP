import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf
import time


it = [1000, 1000, 1000, 1000];
learningRate = 0.01;
fileLinesCount = 1000;
start = time.time()
data, outputs = DataParser.ReadFile("sol100000.txt", fileLinesCount)
end = time.time()
print("Total time elapsed: " + str((end - start) * 1000) + " miliseconds with " + str(fileLinesCount) + " lines of file" )


a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.crossEntropyLoss);

a.pretrain([0.05, 0.05, 0.01, 0.01], [0, 0, 0, 0], data, [0.1, 0.3, 0.5, 0.7], "./summaries/pretrain{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);
c = Classifier.Classifier(a, 14);
c.train(data, outputs, learningRate, 1000);
c.test(data, outputs);

