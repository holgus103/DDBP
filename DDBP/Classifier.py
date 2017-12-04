import tensorflow as tf;
import Tools;

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
        opt = tf.train.AdamOptimizer(learningRate);
        optimizer = opt.minimize(loss);
        self.autoencoder.session.run(tf.variables_initializer([self.weights, self.biases]));
        slot_vars = [self.weights, self.biases] + self.autoencoder.biases + self.autoencoder.weights;
        self.autoencoder.session.run(Tools.initializeOptimizer(opt, slot_vars));
        hist_summaries = [(self.autoencoder.weights[i], 'weights{0}'.format(i)) for i in range(0, len(self.autoencoder.weights))];
        hist_summaries.extend([(self.autoencoder.biases[i], 'biases{0}'.format(i)) for i in range(0, len(self.autoencoder.weights))]);
        summaries = [tf.summary.histogram(v[1], v[0]) for v in hist_summaries];
        summaries.append(tf.summary.scalar("loss", loss));   
        summary_op = tf.summary.merge(summaries);

        writer = tf.summary.FileWriter('./graphs/fine_tuning', graph=self.autoencoder.session.graph_def)
        for i in range(0, it):
            _, summary = self.autoencoder.session.run([optimizer, summary_op], feed_dict={self.inputPlaceholder: data, self.outputPlaceholder: desiredOutput});
            if i % 100 == 0:
                writer.add_summary(summary, i);

        
        





