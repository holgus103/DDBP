
import tensorflow as tf
# based on: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
class Autoencoder:

    def createHiddenLayer(self, prevCount, currCount, input):
            w = tf.Variable(tf.random_normal([prevCount, currCount]));
            b = tf.Variable(tf.random_normal([currCount]));
            self.weights.append(w);
            self.biases.append(b);
            self.layers.append(tf.nn.sigmoid(tf.add(tf.matmul(input, w), b)));

    def __init__(self, inputCount, layerCounts, input):
        self.weights = [];   
        self.biases = [];
        self.layers = [];
        l = len(layerCounts);
        
        self.createHiddenLayer(inputCount, layerCounts[0], input);
        # add encoding layers
        for i in range(1, l - 1):
            self.createHiddenLayer(layerCounts[i-1], layerCounts[i], self.layers[i - 1]);
        
        # add decoding layers
        for i in range(1, l - 2):
            self.createHiddenLayer(layerCounts[l - i - 1], layerCounts[l - i - 2], self.layers[i - 1]);
    