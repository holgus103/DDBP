import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf;

learningRate = 0.01;
dataSet, outputSet = DataParser.ReadFile("D:\\OneDrive\\Uczelnia\\MiNI\\9 semestr\\Sieci neuronowe\\Projekty\\Brydż\\DDBP\\DDBP\\sol100000.txt", 3)
data = dataSet[0]
outputs = outputSet[0]
print (data)
print (outputs)
#data, outputs = DataParser.Parse("AKQT6.3.J876.T43 J875.AJT87.T9.Q6 9.964.AKQ32.K875 432.KQ52.54.AJ92:75755454989842427575");
#l = len(data);
#a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.mseLoss);

#a.pretrain(learningRate, 1000, data);
#c = Classifier.Classifier(a, 14);
#c.train(data, outputs, learningRate, 1000);
