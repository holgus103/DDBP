
import tensorflow as tf
# based on: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
class Autoencoder:
    def mseLoss(pred, actual):
        return tf.reduce_mean(tf.pow(actual - pred, 2));

    def crossEntropyLoss(pred, actual):
        crossEntropy = tf.add(tf.mul(tf.log(pred), actual), tf.mul(tf.log(1 - pred), 1 - actual));
        return -tf.reduce_mean(tf.reduce_sum(crossEntropy, 1));

    def createHiddenLayer(self, prevCount, currCount, input):
            w = tf.Variable(tf.random_normal([prevCount, currCount]));
            b = tf.Variable(tf.random_normal([currCount]));
            self.weights.append(w);
            self.biases.append(b);
            self.layers.append(tf.nn.sigmoid(tf.add(tf.matmul(input, w), b)));

    def __init__(self, inputCount, layerCounts, loss):
        self.loss = loss;
        self.weights = [];   
        self.biases = [];
        self.layers = [];
        self.input = tf.placeholder("float", [None, inputCount]);
        l = len(layerCounts);
        
        self.createHiddenLayer(inputCount, layerCounts[0], self.input);
        # add encoding layers
        for i in range(0, l - 1):
            self.createHiddenLayer(layerCounts[i], layerCounts[i + 1], self.layers[i]);
        
        # add decoding layers
        for i in range(1, l):
            self.createHiddenLayer(layerCounts[l - i], layerCounts[l - i - 1], self.layers[l + i - 2]);
        # add output layer
        self.createHiddenLayer(layerCounts[0], inputCount, self.layers[len(self.layers) - 1]);

    def train(data, desiredOutput, learningRate, it, batchsize,):
        init = tf.global_variables_initializer();

        lossFunction = self.loss(self.layers[len(self.layers) - 1], self.input);
        optimizer = tf.train.RMSPropOptimizer(learningRate).minimize(lossFunction);

        with tf.Session() as session:
             sess.run(init);
            

             for i in range(1, it):
                 sess.run([optimizer, lossFunction], feed_dict={self.input : data});
                
    