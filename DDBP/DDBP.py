import Autoencoder;
import DataParser;
import Classifier;

import tensorflow as tf
import time


it = [10000, 10000, 10000, 1000];
learningRate = 0.01;
fileLinesCount = 3
start = time.time()
data, outputs = DataParser.ReadFile("D:\\OneDrive\\Uczelnia\\MiNI\\9 semestr\\Sieci neuronowe\\Projekty\\Brydż\\DDBP\\DDBP\\sol100000.txt", fileLinesCount)
end = time.time()
print("Total time elapsed: " + str((end - start) * 1000) + " miliseconds with " + str(fileLinesCount) + " lines of file" )

#dataSet, outputSet = DataParser.ReadFile("sol100000.txt", 3)
#data, outputs = DataParser.Parse("AKQT6.3.J876.T43 J875.AJT87.T9.Q6 9.964.AKQ32.K875 432.KQ52.54.AJ92:75755454989842427575");

a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.mseLoss);

a.pretrain(learningRate, it, data);
c = Classifier.Classifier(a, 14);
c.train(data, outputs, learningRate, 10000);
