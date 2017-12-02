import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf;
it = [10000, 10000, 10000, 1000];
learningRate = 0.01;
dataSet, outputSet = DataParser.ReadFile("sol100000.txt", 3)
#data = dataSet[0]
#outputs = outputSet[0]
#print (data)
#print (outputs)
#data, outputs = DataParser.Parse("AKQT6.3.J876.T43 J875.AJT87.T9.Q6 9.964.AKQ32.K875 432.KQ52.54.AJ92:75755454989842427575");
#l = len(data);
a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.mseLoss);

a.pretrain(learningRate, it, dataSet);
c = Classifier.Classifier(a, 14);
c.train(dataSet, outputSet, learningRate, 10000);
