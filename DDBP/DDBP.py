import Autoencoder;
import DataParser;

import tensorflow as tf;

learningRate = 0.01;
data = DataParser.Parse("AKQT6.3.J876.T43 J875.AJT87.T9.Q6 9.964.AKQ32.K875 432.KQ52.54.AJ92");
l = len(data);
a = Autoencoder.Autoencoder(208, [104, 52, 26, 13], Autoencoder.Autoencoder.crossEntropyLoss);
a.pretrain(learningRate, 1000, data.reshape(1, l));
