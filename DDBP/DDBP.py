import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf
import time


it = [1000, 1000, 1000, 1000];
learningRate = 0.01;
fileLinesCount = 10000;
start = time.time()
data, outputs = DataParser.ReadFile("sol100000.txt", fileLinesCount)
end = time.time()
print("Total time elapsed: " + str((end - start) * 1000) + " miliseconds with " + str(fileLinesCount) + " lines of file" )


a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.crossEntropyLoss);

a.pretrain(learningRate, it, data);
c = Classifier.Classifier(a, 14);
c.train(data, outputs, learningRate, 10000);

#config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                        allow_soft_placement=True, device_count = {'CPU': 1})
