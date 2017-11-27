import tensorflow as tf;

class Classifier:
    """description of class"""
    def __init__(self, autoencoder, outputs):
        self.autoencoder = autoencoder;
        self.inputPlaceholder = tf.placeholder("float", [None, self.autoencoder.inputCount]);
        self.encoder = autoencoder.buildCompleteNet(self.inputPlaceholder);
        input = self.encoder[len(self.encoder) - 1];
        self.weights = tf.Variable(tf.random_normal([input.shape[1].value, outputs]));
        self.biases = tf.Variable(tf.random_normal([outputs]));
        self.layer = tf.nn.softmax(tf.matmul(input, self.weights) + self.biases);
        self.outputPlaceholder = tf.placeholder("float", [None, outputs]);
        

    def train(self, data, desiredOutput, learningRate, it):
        loss = tf.reduce_mean(tf.pow(self.layer - self.outputPlaceholder, 2));
        optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss);
        self.autoencoder.session.run(tf.initialize_variables([self.weights, self.biases]));
        for i in range(0, it):
            self.autoencoder.session.run([optimizer, loss], feed_dict={self.inputPlaceholder: data, self.outputPlaceholder: desiredOutput});

        
        





