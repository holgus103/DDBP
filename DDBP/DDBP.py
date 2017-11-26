import Autoencoder;
import tensorflow as tf;

learningRate = 0.01;
a = Autoencoder.Autoencoder(10, [8,2], Autoencoder.Autoencoder.crossEntropyLoss);
a.pretrain(learningRate);
