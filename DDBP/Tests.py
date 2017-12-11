import DataParser;
import Autoencoder;
import Classifier;
import tensorflow as tf;
import time;



def one_layer_cross_entropy_rms_opttest() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain([learningRate], [0], data, [0.0001], "./summaries/one_layer_smalldataset_rms_crsent_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);


def one_layer_many_pretrains() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain([learningRate], [0], data, [1.0], "./summaries/one_layer_many_pretrains-part1{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);
    learningRate = 0.01;
    a.pretrain([learningRate], [0], data, [0.0001], "./summaries/one_layer_many_pretrains-part2{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def one_layer_rms_rms_opttest() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.mseLoss);
    a.pretrain([learningRate], [0], data, [0], "./summaries/one_layer_smalldataset_rms_rms_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def one_layer_direct_rms_opttest() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.directError);
    a.pretrain([learningRate], [0], data, [0], "./summaries/one_layer_smalldataset_direct_rms_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);



def one_layer_cross_entropy_gradient_descent_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain([learningRate], [0], data, [1.0], "./summaries/one_layer_smalldataset_gradient_crsent{0}".format(learningRate) + "{0}", tf.train.GradientDescentOptimizer);

def one_layer_cross_entropy_adam_opt_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain([learningRate], [0], data, [1.0], "./summaries/one_layer_smalldataset_adam_crsent{0}".format(learningRate) + "{0}", tf.train.AdamOptimizer);



def one_layer_rms_error_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.mseLoss);
    a.pretrain([learningRate], [0], data, [1.0], "./summaries/one_layer_smalldataset_rms_rmserror{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def full_cross_entropy_test() :
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain([0.01, 0.01, 0.01, 0.01], [0, 0, 0, 0], data, [0.0001, 0.01, 5, 10], "./summaries/all_layers_smalldataset_rms_crsent{0}" , tf.train.RMSPropOptimizer);

def save_test():
    data, outputs = DataParser.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = Autoencoder.Autoencoder(217, [104], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain([learningRate], [0], data, [0.0001], "./summaries/save_test{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);
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

def smalldataset_goodresults():
    data, outputs = DataParser.ReadFile("sol100000.txt", 60);
    a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.crossEntropyLoss);
    a.pretrain([0.01, 0.01, 0.01, 0.01], [5000, 3000, 4000, 4000], data[0:1001], [0.0001, 0.01, 5, 10], "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer);
    c = Classifier.Classifier(a, 14);
    c.train(data[0:1001], outputs[0:1001], 0.01, 10000, "./summaries/finetuning");
    c.test(data[1001: 1200], outputs[1001:1200]);
    c.save_model("smalldata_impressive results")

def smalldataset_goodresults_restore():
    data, outputs = DataParser.ReadFile("sol100000.txt", 60);
    a = Autoencoder.Autoencoder(217, [104, 52, 26, 13], Autoencoder.Autoencoder.crossEntropyLoss);
    c = Classifier.Classifier(a, 14);
    saver = tf.train.Saver();
    saver.restore(a.session, "./models/smalldata_impressive results");
    print(c.test(data[1001: 1200], outputs[1001:1200]));
    

    