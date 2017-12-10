import DataParser;
import Autoencoder;
import tensorflow as tf;
import time;



def one_layer_cross_entropy_rms_opttest() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain(learningRate, [0.1], data, [0.0001], "./summaries/one_layer_smalldataset_rms_crsent_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def one_layer_rms_rms_opttest() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.mseLoss);
    a.pretrain(learningRate, [0], data, [0], "./summaries/one_layer_smalldataset_rms_rms_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def one_layer_direct_rms_opttest() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.directError);
    a.pretrain(learningRate, [0], data, [0], "./summaries/one_layer_smalldataset_direct_rms_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);



def one_layer_cross_entropy_gradient_descent_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain(learningRate, [0], data, [1.0], "./summaries/one_layer_smalldataset_gradient_crsent{0}".format(learningRate) + "{0}", tf.train.GradientDescentOptimizer);

def one_layer_cross_entropy_adam_opt_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain(learningRate, [0], data, [1.0], "./summaries/one_layer_smalldataset_adam_crsent{0}".format(learningRate) + "{0}", tf.train.AdamOptimizer);



def one_layer_rms_error_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.mseLoss);
    a.pretrain(learningRate, [0], data, [1.0], "./summaries/one_layer_smalldataset_rms_rmserror{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def full_cross_entropy_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain(learningRate, [0], data, [0.1, 0.3, 0.5, 0.7], "./summaries/all_layers_smalldataset_rms_crsent{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def save_test():
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain(learningRate, [0], data, [0.0001], "./summaries/one_layer_smalldataset_rms_crsent_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);
    saver = tf.train.Saver();
    saver.save(a.session, "./models/save_test.ckpt".format(time.ctime().replace(" ", "_")));

def restore_test():
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    net = a.buildPretrainNet(0, a.input);
    loss_function = Autoencoder.Autoencoder.crossEntropyLoss(net[len(net) - 1], a.input);
    saver = tf.train.Saver();
    saver.restore(a.session, "./models/save_test.ckpt")
    lval = a.session.run([loss_function], feed_dict={a.input : data});
    print(lval);
    