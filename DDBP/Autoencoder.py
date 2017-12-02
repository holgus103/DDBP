import tensorflow as tf;
# based on: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
class Autoencoder:
    def mseLoss(pred, actual):
        return tf.reduce_mean(tf.pow(actual - pred, 2));

    def crossEntropyLoss(pred, actual):
        p = tf.convert_to_tensor(pred);
        a = tf.convert_to_tensor(actual);
        crossEntropy = tf.add(tf.multiply(tf.log(p), a), tf.multiply(tf.log(1 - p), 1 - a));
        return -tf.reduce_mean(tf.reduce_sum(crossEntropy, 1));

    @property
    def session(self):
        return self.__session;

    def createLayer(self, index, input, isFixed = False, isDecoder = False):
        if isFixed:
            if isDecoder:
                return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixedWeights[index], transpose_b = isDecoder), self.outBiasesFixed[index]));
            return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixedWeights[index], transpose_b = isDecoder), self.biases[index]));
        if isDecoder:
            return tf.nn.sigmoid(tf.add(tf.matmul(input, self.weights[index], transpose_b = isDecoder), self.outBiasesFixed[index]));
        return tf.nn.sigmoid(tf.add(tf.matmul(input, self.weights[index], transpose_b = isDecoder), self.biases[index]));

    def createWeights(self, prevCount, currCount):
            w = tf.Variable(tf.random_normal([prevCount, currCount]), trainable = True, name='v_W{0}'.format(currCount));
            b = tf.Variable(tf.random_normal([currCount]), trainable = True, name='v_B{0}'.format(currCount));
            self.weights.append(w);
            self.biases.append(b);
            w_f = tf.Variable(tf.identity(w), trainable = False, name='f_W{0}'.format(currCount));
            b_f = tf.Variable(tf.identity(b), trainable = False, name='f_B{0}'.format(currCount));
            self.fixedWeights.append(w_f);
            self.fixedBiases.append(b_f);
            
            b_out = tf.Variable(tf.random_normal([prevCount]), trainable = True, name='v_B_out{0}'.format(currCount));
            b_out_fixed = tf.Variable(tf.identity(b_out), trainable = False, name='f_B_out{0}'.format(currCount));
            self.outBiases.append(b_out);
            self.outBiasesFixed.append(b_out_fixed);
            #self.layers.append(tf.nn.sigmoid(tf.add(tf.matmul(input, w), b)));

    def __init__(self, inputCount, layerCounts, loss):
        self.loss = loss;
        self.inputCount = inputCount;
        self.layerCounts = layerCounts;
        self.weights = [];   
        self.biases = [];
        self.outBiases = [];
        self.outBiasesFixed = [];
        #self.layers = [];
        self.fixedWeights = [];
        self.fixedBiases = []
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

    def getVariablesToInit(self, n):
        vars = [self.weights[n], self.biases[n]]

        vars.append(self.outBiases[n]);

        if 0<n:
            vars.append(self.fixedBiases[n-1]);
            vars.append(self.fixedWeights[n-1]);
            vars.append(self.outBiasesFixed[n-1]);
        return vars;

     
    def pretrain(self, learningRate, it, data):
        self.__session = tf.Session();
        init = tf.global_variables_initializer();
        self.session.run(init);
        for i in range(0, len(self.layerCounts)):
            #with tf.Graph().as_default() as g:
            input = tf.placeholder("float", [len(data), self.inputCount]);
            net = self.buildPretrainNet(i, input);
            lossFunction = self.loss(net[len(net) - 1], input);
            optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(lossFunction);
            loss_summary = tf.summary.scalar("loss", lossFunction);
            weights_summary = tf.summary.histogram("weights", self.weights[i]);
            biases_summary = tf.summary.histogram("biases", self.biases[i]);
            summary_op = tf.summary.merge([loss_summary, weights_summary, biases_summary]);
            writer = tf.summary.FileWriter('./graphs/pretraining_{0}'.format(i), graph=self.session.graph_def, flush_secs = 10000);
            self.session.run(tf.initialize_variables(self.getVariablesToInit(i)));         
            for j in range(1, it[i]):
                    _, summary = self.session.run([optimizer, summary_op], feed_dict={input : data});
                    if j % 100 == 0:
                        writer.add_summary(summary, j);
            
    def buildCompleteNet(self, input):
        net = [];
        inp = input;
        for i in range(0, len(self.weights)):
            inp = self.createLayer(i, inp);
            net.append(inp);
            
        return net;

    
    def buildPretrainNet(self, n, input):
        
        layers = [];
        inp = input;
        for i in range(0, n):
            inp = self.createLayer(i, inp, isFixed = True);
            layers.append(inp);
        
        inp = self.createLayer(n, inp);
        layers.append(inp);

        inp = self.createLayer(n, inp, isDecoder = True);
        layers.append(inp);
        
        for i in range(0, n):
            inp = self.createLayer(n - 1 - i, inp, isFixed = True, isDecoder = True);
            layers.append(inp);
        return layers;
            
            
        
                
    