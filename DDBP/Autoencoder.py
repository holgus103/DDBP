
import tensorflow as tf
# based on: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
class Autoencoder:
    def mseLoss(pred, actual):
        return tf.reduce_mean(tf.pow(actual - pred, 2));

    def crossEntropyLoss(pred, actual):
        crossEntropy = tf.add(tf.mul(tf.log(pred), actual), tf.mul(tf.log(1 - pred), 1 - actual));
        return -tf.reduce_mean(tf.reduce_sum(crossEntropy, 1));

    def createLayer(self, index, input, isFixed = False, isDecoder = False):
        if isFixed:
            if isDecoder:
                return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixedWeights[index], transpose_b = isDecoder), self.outBiasesFixed[index]));
            return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixedWeights[index], transpose_b = isDecoder), self.fixedBiases[index]));
        if isDecoder:
            return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixedWeights[index], transpose_b = isDecoder), self.outBiasesFixed[index]));
        return tf.nn.sigmoid(tf.add(tf.matmul(input, self.weights[index], transpose_b = isDecoder), self.biases[index]));

    def createWeights(self, prevCount, currCount):
            w = tf.Variable(tf.random_normal([prevCount, currCount]), trainable = True);
            b = tf.Variable(tf.random_normal([currCount]), trainable = True);
            self.weights.append(w);
            self.biases.append(b);
            w_f = tf.Variable(tf.random_normal([prevCount, currCount]), trainable = False);
            b_f = tf.Variable(tf.random_normal([currCount]), trainable = False);
            self.fixedWeights.append(w_f);
            self.fixedBiases.append(b_f);
            
            b_out = tf.Variable(tf.random_normal([prevCount]), trainable = True);
            b_out_fixed = tf.Variable(tf.random_normal([prevCount]), trainable = False);
            self.outBiases.append(b_out);
            self.outBiasesFixed.append(b_out_fixed);
            #self.layers.append(tf.nn.sigmoid(tf.add(tf.matmul(input, w), b)));

    def __init__(self, inputCount, layerCounts, loss):
        self.loss = loss;
        self.layerCounts = layerCounts;
        self.weights = [];   
        self.biases = [];
        self.outBiases = [];
        self.outBiasesFixed = [];
        #self.layers = [];
        self.fixedWeights = [];
        self.fixedBiases = []
        self.input = tf.placeholder("float", [None, inputCount]);
        l = len(layerCounts);
        
        self.createWeights(inputCount, layerCounts[0]);
        # add encoding layers
        for i in range(0, l - 1):
            self.createWeights(layerCounts[i], layerCounts[i + 1]);
        
        # add decoding layers
        #for i in range(1, l):
        #    self.createWeights(layerCounts[l - i], layerCounts[l - i - 1]);
        # add output layer
        #self.createWeights(layerCounts[0], inputCount);

    def train(data, desiredOutput, learningRate, it, batchsize,):
        init = tf.global_variables_initializer();

        lossFunction = self.loss(self.layers[len(self.layers) - 1], self.input);
        optimizer = tf.train.RMSPropOptimizer(learningRate).minimize(lossFunction);

        with tf.Session() as session:
             sess.run(init);
            

             for i in range(1, it):
                 sess.run([optimizer, lossFunction], feed_dict={self.input : data});
    
    def pretrain(self):
        for i in range(0, len(self.layerCounts)):
            net = self.buildPretrainNet(i)

    def buildPretrainNet(self, n):
        
        layers = [];
        inp = self.input;
        for i in range(0, n):
            inp = self.createLayer(i, inp, isFixed = True);
            layers.append(inp);
        
        inp = self.createLayer(n, inp);
        layers.append(inp);

        inp = self.createLayer(n, inp, isDecoder = True);
        layers.append(inp);
        
        for i in range(0, n):
            inp = self.createLayer(n - i, inp, isFixed = True, isDecoder = True);
            layers.append(inp);
        return layers;
            
            
        
                
    