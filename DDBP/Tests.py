import DataParser;
import Autoencoder;
import tensorflow as tf;



def one_layer_cross_entropy_rms_opttest() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain(learningRate, 0, data, [0.1], "./summaries/one_layer_smalldataset_rms_crsent_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);


def one_layer_cross_entropy_gradient_descent_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain(learningRate, 0, data, [1.0], "./summaries/one_layer_smalldataset_gradient_crsent{0}".format(learningRate) + "{0}", tf.train.GradientDescentOptimizer);

def one_layer_cross_entropy_adam_opt_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain(learningRate, 0, data, [1.0], "./summaries/one_layer_smalldataset_adam_crsent{0}".format(learningRate) + "{0}", tf.train.AdamOptimizer);



def one_layer_rms_error_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.mseLoss);
    a.pretrain(learningRate, 0, data, [1.0], "./summaries/one_layer_smalldataset_rms_rmserror{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def full_cross_entropy_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain(learningRate, 0, data, [0.1, 0.3, 0.5, 0.7], "./summaries/all_layers_smalldataset_rms_crsent{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);