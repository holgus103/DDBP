import models;
import tensorflow as tf;
import time;
import random;
import math;
import models;
import data_parser as dp;


def one_layer_cross_entropy_rms_opttest() :
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = models.Autoencoder(217, [104], models.Model.cross_entropy_loss);
    a.pretrain(learningRate, 0, 0, data, 0.0001, 0.01, "./summaries/one_layer_smalldataset_rms_crsent_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);


def one_layer_many_pretrains() :
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = models.Autoencoder(217, [104], models.Model.cross_entropy_loss);
    a.pretrain(learningRate, 0, 0, data, 1.0, 0.01, "./summaries/one_layer_many_pretrains-part1{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);
    learningRate = 0.01;
    a.pretrain(learningRate, 0, 0, data, 0.0001, 0.01, "./summaries/one_layer_many_pretrains-part2{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def one_layer_rms_rms_opttest() :
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = models.Autoencoder(217, [104], models.Autoencoder.mseLoss);
    a.pretrain(learningRate, 0, 0, data, 0, 0.01, "./summaries/one_layer_smalldataset_rms_rms_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);

def one_layer_direct_rms_opttest() :
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = models.Autoencoder(217, [104], models.Autoencoder.directError);
    a.pretrain(learningRate, 0, 0, data, 0, 0.01, "./summaries/one_layer_smalldataset_direct_rms_{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer);



def one_layer_cross_entropy_gradient_descent_test() :
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = models.Autoencoder(217, [104], models.Model.cross_entropy_loss);
    a.pretrain(learningRate, 0, 0, data, 1.0, 0.01, "./summaries/one_layer_smalldataset_gradient_crsent{0}".format(learningRate) + "{0}", tf.train.GradientDescentOptimizer);

def one_layer_cross_entropy_adam_opt_test() :
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = models.Autoencoder(217, [104], models.Model.cross_entropy_loss);
    a.pretrain(learningRate, 0, 0, data, 1.0, 0.01, "./summaries/one_layer_smalldataset_adam_crsent{0}".format(learningRate) + "{0}", tf.train.AdamOptimizer);



def one_layer_rms_error_test() :
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    learningRate = 0.01;
    a = models.Autoencoder(217, [104], models.Autoencoder.mseLoss);
    a.pretrain(learningRate, 0,  0, data, 1.0, 0.01, "./summaries/one_layer_smalldataset_rms_rmserror{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer, 0.2);

def full_cross_entropy_test() :
    path = "./summaries/all_layers_smalldataset_rms_crsent{0}";
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    a = models.Autoencoder(217, [104, 52, 26, 13], models.Model.cross_entropy_loss);
    a.pretrain(0.01, 0, 5000, data, 0.0001, 0.01, path, tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 1, 3000, data, 0.01, 0.01, path, tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 2, 4000, data, 5, 0.01, path , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 3, 4000, data, 10, 0.01, path , tf.train.RMSPropOptimizer, 0.2);

def save_test():
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = models.Autoencoder(217, [104], models.Model.cross_entropy_loss);
    a.pretrain(learningRate, 0, 0, data, 0.0001, 0.01, "./summaries/save_test{0}".format(learningRate) + "{0}", tf.train.RMSPropOptimizer, 0.2);
    saver = tf.train.Saver();
    saver.save(a.session, "./models/save_test.ckpt".format(time.ctime().replace(" ", "_")));

def restore_test():
    data, outputs = dp.ReadFile("sol100000.txt", 50);
    learningRate = 0.05;
    a = models.Autoencoder(217, [104], models.Model.cross_entropy_loss);
    net = a.buildPretrainNet(0, a.input);
    loss_function = models.Model.cross_entropy_loss(net[len(net) - 1], a.input);
    saver = tf.train.Saver();
    saver.restore(a.session, "./models/save_test.ckpt")
    lval = a.session.run([loss_function], feed_dict={a.input : data});
    print(lval);

def smalldataset_goodresults():
    data, outputs = dp.ReadFile("sol100000.txt", 60);
    a = models.Autoencoder(217, [104, 52, 26, 13], models.Model.cross_entropy_loss);
    a.pretrain(0.01, 0, 5000, data[0:1001], 0, 0.01, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 1, 3000, data[0:1001], 0, 0.01, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 2, 4000, data[0:1001], 0, 0.01, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 3, 4000, data[0:1001], 0, 0.01, "./summaries/pretrain{0}" , tf.train.RMSPropOptimizer, 0.2);
    c = models.Classifier(a, 14);
    c.train(data[0:1001], outputs[0:1001], 0.01, 10000, 0.01, "./summaries/finetuning");
    c.test(data[1001: 1200], outputs[1001:1200]);
    c.save_model("smalldata_impressive results")

def smalldataset_goodresults_restore():
    data, outputs = dp.ReadFile("sol100000.txt", 60);
    a = models.Autoencoder(217, [104, 52, 26, 13], models.Model.cross_entropy_loss);
    c = Classifier.Classifier(a, 14);
    saver = tf.train.Saver();
    saver.restore(a.session, "./models/smalldata_impressive results");
    print(c.test(data[1001: 1200], outputs[1001:1200]));
    
def variousL1Sizes():
    # 1.1 to 2.0 divisors
    sizes = [197, 180, 166, 155, 144, 135, 127, 120, 114, 108]
    data, outputs = dp.ReadFile("sol100000.txt", 10000);
    #l = len(data);
    #splitIndex = math.floor(data * 2/3); 
    #testData = data[0 : splitIndex];
    #trainData = data[splitIndex + 1 : l];
    #testOutput = outputs[0:splitIndex];
    #trainOutput = outputs[splitIndex + 1 : l];
    for i in sizes:
        a = models.Autoencoder(217, [i], models.Model.cross_entropy_loss);
        a.pretrain(0.01, 0, 10000, data, 0, 0.01, "./summaries/oneLayer{0}".format(i) + "{0}" , tf.train.RMSPropOptimizer, 0.2);

def fulltest():
    data, outputs = dp.ReadFile("sol100000.txt", 1500);
    path = "./summaries/finaltest1500_7laters/{0}";
    a = models.Autoencoder(217, [173, 140, 110, 90, 56, 45, 36], models.Model.cross_entropy_loss);
    a.pretrain(0.01, 0, 4000, data[0:20000], 0, 0.01, path , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 1, 10000, data[0:20000], 0, 0.01, path , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 2, 10000, data[0:20000], 0, 0.01, path , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 3, 10000, data[0:20000], 0, 0.01, path , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 4, 10000, data[0:20000], 0, 0.01, path , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 5, 10000, data[0:20000], 0, 0.01, path , tf.train.RMSPropOptimizer, 0.2);
    a.pretrain(0.01, 6, 10000, data[0:20000], 0, 0.01, path , tf.train.RMSPropOptimizer, 0.2);
    c = models.Classifier(a, 14);
    c.train(data[0:20000], outputs[0:20000], 0.01, 10000, 0.01, "./summaries/finaltest1500_7laters/finetuning");
    print(c.test(data[0:20000], outputs[0:20000]));
    print(c.test(data[20001: 30000], outputs[20001:30000]));

    