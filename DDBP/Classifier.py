import tensorflow as tf;

class Classifier:
    """description of class"""
    def __init__(self, autoencoder, outputs):
        self.autoencoder = autonencoder;
        self.inputPlaceholder = tf.placeholder("float", [None, self.autoencoder.inputCount]);
        self.encoder = autoencoder.buildCompleteNet(self.inputPlaceholder);
        input = encoder[len(encoder) - 1];
        self.weights = tf.Variable(tf.random_normal([input.shape[1].value, outputs]));
        self.biases = tf.Variable(tf.random_normal([input.shape[1].value));
        self.layer = tf.nn.softmax(tf.matmul(input, self.weights) + self.biases);
        self.outputPlaceholder = tf.placeholder("float", [None, outputs]);
        

    def train(self, data, desiredOutput, learningRate, it):
        loss = tf.reduce_mean(tf.pow(self.layer - desiredOutput, 2));
        optimizer = tf.train.RMSPropOptimizer(learningRate).minimize(lossFunction);
        self.autoencoder.session.run(tf.initialize_variables([self.weights, self.layer, self.biases, loss, optimizer]));
        for i in range(0, it):
            self.autoencoder.session.run([optimizer, lossFunction], feed_dict={self.inputPlaceholder: data, self.outputPlaceholder: desiredOutput});

        
        





