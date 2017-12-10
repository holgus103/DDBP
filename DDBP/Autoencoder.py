import tensorflow as tf;
import Tools
import numpy;
# based on: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
class Autoencoder:
    def mseLoss(pred, actual):
        return tf.reduce_mean(tf.pow(actual - pred, 2));

    def crossEntropyLoss(pred, actual):
        p = tf.convert_to_tensor(pred);
        a = tf.convert_to_tensor(actual);
        crossEntropy = tf.add(tf.multiply(tf.log(p + 1e-10), a), tf.multiply(tf.log(1 - p + 1e-10), 1 - a));
        return -tf.reduce_mean(tf.reduce_sum(crossEntropy, 1));


    def directError(pred, actual):
        return tf.reduce_mean(tf.abs(actual - pred));

    @property
    def session(self):
        return self.__session;

    def createLayer(self, index, input, isFixed = False, isDecoder = False):
        if isFixed:
            if isDecoder:
                return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixedWeights[index], transpose_b = isDecoder), self.outBiasesFixed[index]));
            return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixedWeights[index], transpose_b = isDecoder), self.fixedBiases[index]));
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
        self.input = tf.placeholder("float", [None, self.inputCount]);
        #self.layers = [];
        self.fixedWeights = [];
        self.fixedBiases = []
        l = len(layerCounts);
        self.prepare_session();
        self.createWeights(inputCount, layerCounts[0]);
        # add encoding layers
        for i in range(0, l - 1):
            self.createWeights(layerCounts[i], layerCounts[i + 1]);
        
        init = tf.global_variables_initializer();
        self.session.run(init);

    def getVariablesToInit(self, n):
        vars = [self.weights[n], self.biases[n]]

        vars.append(self.outBiases[n]);

        if 0<n:
            vars.append(self.fixedBiases[n-1]);
            vars.append(self.fixedWeights[n-1]);
            vars.append(self.outBiasesFixed[n-1]);
        return vars;

    def prepare_session(self):
        config = tf.ConfigProto(inter_op_parallelism_threads=4,intra_op_parallelism_threads=4);
        self.__session = tf.Session(config=config);

     
    def pretrain(self, learningRate, it, data, ep, summary_path, optimizer_class):
        "Please remember that the summary_path must contain one argument for formatting"
        for i in range(0, len(self.layerCounts)):
            #with tf.Graph().as_default() as g:
            input = self.input;
            net = self.buildPretrainNet(i, input);
            loss_function = self.loss(net[len(net) - 1], input);
            opt = optimizer_class(learningRate[i]);
            optimizer = opt.minimize(loss_function);    
            vars = self.getVariablesToInit(i);
            #self.session.run(tf.variables_initializer(vars));  
            self.session.run(Tools.initializeOptimizer(opt, vars));
            loss_summary = tf.summary.scalar("loss", loss_function);
            weights_summary = tf.summary.histogram("weights", self.weights[i]);
            biases_summary = tf.summary.histogram("biases", self.biases[i]);
            summary_op = tf.summary.merge([loss_summary, weights_summary, biases_summary]);
            writer = tf.summary.FileWriter(summary_path.format(i) + (ep[i] > 0 and "ep{0}".format(ep[i]) or "it{0}".format(it[i])) , graph=self.session.graph, flush_secs = 10000);
            
            if(it[i] > 0):
                for j in range(1, it[i]):
                        _, summary = self.session.run([optimizer, summary_op], feed_dict={input : data});
                        if j % 100 == 0:
                            print("pretraining {0} - it {1}".format(i, j));
                            writer.add_summary(summary, j);
            else:
                j = 0;
                while True:
                    _, summary, lval = self.session.run([optimizer, summary_op, loss_function], feed_dict={input : data});
                    if j % 100 == 0:
                        print("pretraining {0} - it {1} - lval {2}".format(i, j, lval));
                        writer.add_summary(summary, j);
                    j = j + 1;
                    if(lval <= ep[i]):
                        print("pretraining ended {0} - it {1} - lval {2}".format(i, j, lval));
                        #print(numpy.sum(numpy.subtract(self.session.run([net[length(net) - 1]], feed_dict={input : [data[0]]})[0] - data[0])));
                        break;
            

            
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


            
            
        
                
    