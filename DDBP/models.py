import tensorflow as tf;
import time;

class Model:
    """
    Base class for models containing static methods

    """
    def mse_loss(pred, actual):
        """
        Mean Squared Error loss function 

        Parameters
        ----------
        pred : Tensor
            Tensor containing the network's output 
        actual : Autoencoder
            Tensor containing the desired output

        Returns
        -------
        Tensor 
            Tensor used to calculate the error's value
        """
        return tf.reduce_mean(tf.pow(actual - pred, 2));

    def cross_entropy_loss(pred, actual):
        """
        Cross Entropy loss function 

        Parameters
        ----------
        pred : Tensor
            Tensor containing the network's output 
        actual : Autoencoder
            Tensor containing the desired output

        Returns
        -------
        Tensor 
            Tensor used to calculate the error's value
        """
        p = tf.convert_to_tensor(pred);
        a = tf.convert_to_tensor(actual);
        crossEntropy = tf.add(tf.multiply(tf.log(p + 1e-10), a), tf.multiply(tf.log(1 - p + 1e-10), 1 - a));
        return -tf.reduce_mean(tf.reduce_sum(crossEntropy, 1));


    def direct_error(pred, actual):
        """
        Direct loss function 

        Parameters
        ----------
        pred : Tensor
            Tensor containing the network's output 
        actual : Autoencoder
            Tensor containing the desired output

        Returns
        -------
        Tensor 
            Tensor used to calculate the error's value
        """
        return tf.reduce_mean(tf.abs(actual - pred));

    def initialize_optimizer(opt, vars):
        """
        Initializes all optimizer slots

        Parameters
        ----------
        opt : Optimizer
            Optimizer that needs initialization
        vars : Autoencoder
            Variables used

        Returns
        -------
        Operation 
            Initialization operation
        """
        to_init = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in vars];
        if(type(opt) is tf.train.AdamOptimizer):
            to_init.extend(s for s in list(opt._get_beta_accumulators()) if s is not None);    
        
        return tf.variables_initializer([s for s in to_init if s is not None]);

class Autoencoder(Model):
    """
    Class implementing a multilayered Autoencoder Network
    Reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

    """


    @property
    def session(self):
        """
        Model's Tensorflow session

        Parameters
        ----------
        self : Autoencoder

        Returns
        -------
        tensorflow.Session 
            Session currently used by the model
        """
        return self.__session;

    def create_layer(self, index, input, is_fixed = False, is_decoder = False):
        """
        Creates an autoencoder layer

        Parameters
        ----------
        self : Autoencoder
        index : int
            Layer index
        input : Tensor
            Previous layer feed to the new layer
        is_fixed : bool
            Boolean value indicating whether the layer needs to be temporarily frozen
        is_decoder : bool
            Boolean value indicating if the layer needs to serve as an encoder or a decoder
        Returns
        -------
        Tensor 
            Created layer
        """
        if is_fixed:
            if is_decoder:
                return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixed_weights[index], transpose_b = is_decoder), self.out_biases_fixed[index]));
            return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixed_weights[index], transpose_b = is_decoder), self.fixed_biases[index]));
        if is_decoder:
            return tf.nn.sigmoid(tf.add(tf.matmul(input, self.weights[index], transpose_b = is_decoder), self.out_biases[index]));
        return tf.nn.sigmoid(tf.add(tf.matmul(input, self.weights[index], transpose_b = is_decoder), self.biases[index]));

    def create_weights(self, prev_count, curr_count):
        """
        Creates weights for a layer

        Parameters
        ----------
        self : Autoencoder
        prev_count : int
            Neuron count of the previous layer
        curr_count : Tensor
            Neuron count of the current layer

        """
        w = tf.Variable(tf.random_normal([prev_count, curr_count]), trainable = True, name='v_W{0}'.format(curr_count));
        b = tf.Variable(tf.random_normal([curr_count]), trainable = True, name='v_B{0}'.format(curr_count));
        self.weights.append(w);
        self.biases.append(b);
        w_f = tf.Variable(tf.identity(w), trainable = False, name='f_W{0}'.format(curr_count));
        b_f = tf.Variable(tf.identity(b), trainable = False, name='f_B{0}'.format(curr_count));
        self.fixed_weights.append(w_f);
        self.fixed_biases.append(b_f);
        
        b_out = tf.Variable(tf.random_normal([prev_count]), trainable = True, name='v_B_out{0}'.format(curr_count));
        b_out_fixed = tf.Variable(tf.identity(b_out), trainable = False, name='f_B_out{0}'.format(curr_count));
        self.out_biases.append(b_out);
        self.out_biases_fixed.append(b_out_fixed);

    def __init__(self, input_count, layer_counts, loss):
        """
        Main class constructor

        Parameters
        ----------
        self : Autoencoder
        input_count : int
            Number of network's inputs
        layer_counts : list
            List of neuron counts for each layer
        loss : Tensor
            Loss function used during model optimalization

        """
        self.loss = loss;
        self.input_count = input_count;
        self.layer_counts = layer_counts;
        self.weights = [];   
        self.biases = [];
        self.out_biases = [];
        self.out_biases_fixed = [];
        self.input = tf.placeholder("float", [None, self.input_count]);
        self.fixed_weights = [];
        self.fixed_biases = []
        l = len(layer_counts);
        self.prepare_session();
        self.create_weights(input_count, layer_counts[0]);
        # add encoding layers
        for i in range(0, l - 1):
            self.create_weights(layer_counts[i], layer_counts[i + 1]);
        
        init = tf.global_variables_initializer();
        self.session.run(init);

    def get_variables_to_init(self, n):
        """
        Creates a list of variables that need to be currently initialized

        Parameters
        ----------
        self : Autoencoder
        n : int
            Layer index

        Returns
        -------
        list 
            Returns a list of tensorflow.Variable objects that neeed to be intialized during this step

        """

        vars = [];

        if 0<n:
            vars.append(self.fixed_biases[n-1]);
            vars.append(self.fixed_weights[n-1]);
            vars.append(self.out_biases_fixed[n-1]);
        return vars;

    def prepare_session(self):
        """
        Prepares a new session for the model

        Parameters
        ----------
        self : Autoencoder

        """
        config = tf.ConfigProto();
        self.__session = tf.Session(config=config);

     
    def pretrain(self, learning_rate, i, it, data, ep, delta, summary_path, optimizer_class = tf.train.RMSPropOptimizer, m = 0.2, decay = 0.9, no_improvement = 5):
        """
        Pretrains one layer with specified parameters
        Please remember that the summary_path must contain one argument for formatting

        Parameters
        ----------
        self : Autoencoder
        learning_rate : int
            Learning rate for the optimizer
        i : int
            Layer index
        it : int
            Number of iterations used
        data : list
            List of numpy arrays feed to the input placeholder used for training
        delta : float
            If the improvement between epochs is smaller than delta, the training process is aborted
        ep : float
            If it is 0, then the training will be executed until the error value is larger than ep
        summary_path : string
            Path used to store Tensorflow summaries generated by the model during training
        optimizer_class : float
            Optimized class used during pretraining
        m : float
            Momentum
        decay : float
            Learning rate decay
        no_improvement : int
            If the error function value worsens the amount of times specified by this parameter the calculation will be aborted

        """

        input = self.input;
        step = tf.Variable(0, name='global_step', trainable=False);
        net = self.build_pretrain_net(i, input);
        loss_function = self.loss(net[len(net) - 1], input);
        if(optimizer_class is tf.train.GradientDescentOptimizer):
            opt = optimizer_class(learning_rate);
        else:
            opt = optimizer_class(learning_rate, momentum = m);
        optimizer = opt.minimize(loss_function, global_step=step);    
        vars = self.get_variables_to_init(i);
        vars.append(step);
        self.session.run(tf.variables_initializer(vars));  
        vars.extend([self.weights[i], self.biases[i], self.out_biases[i]])
        self.session.run(Model.initialize_optimizer(opt, vars));
        loss_summary = tf.summary.scalar("loss", loss_function);
        weights_summary = tf.summary.histogram("weights", self.weights[i]);
        biases_summary = tf.summary.histogram("biases", self.biases[i]);
        summary_op = tf.summary.merge([loss_summary, weights_summary, biases_summary]);
        writer = tf.summary.FileWriter(summary_path.format(i) + (ep > 0 and "ep{0}".format(ep) or "it{0}".format(it)) , graph=self.session.graph, flush_secs = 10000);

        if(delta > 0):
            no_improvement_counter = 0;
            prev_val = 0;
            it_counter = 0;
            while True:
                for k in range(0, len(data)):
                    lval, _, summary = self.session.run([loss_function, optimizer, summary_op], feed_dict={input : data[k]});
                    if it_counter % 100 == 0:
                        print("pretraining {0} - it {1} - lval {2}".format(i, it_counter, lval));
                        writer.add_summary(summary, it_counter);
                        # no significant change
                        if prev_val != 0 and (prev_val - lval) < delta:
                            if(no_improvement_counter > no_improvement):
                                print("terminating due to no improvement");
                                print("pretraining {0} - it {1} - lval {2}".format(i, it_counter, lval));
                                return
                            else:
                                no_improvement_counter = no_improvement_counter + 1;
                        prev_val = lval;
                    it_counter = it_counter + 1;


        elif it > 0:
            for j in range(1, it):
                    for k in range(0, len(data)):
                        lval, _, summary = self.session.run([loss_function, optimizer, summary_op], feed_dict={input : data[k]});
                    if j % 100 == 0:
                        print("pretraining {0} - it {1} - lval {2}".format(i, j, lval));
                        writer.add_summary(summary, j);
        else:
            j = 0;
            while True:
                for k in range(0, len(data)):
                    _, summary, lval = self.session.run([optimizer, summary_op, loss_function], feed_dict={input : data[k]});
                
                if j % 100 == 0:
                    print("pretraining {0} - it {1} - lval {2}".format(i, j, lval));
                    writer.add_summary(summary, j);
                j = j + 1;
                if(lval <= ep):
                    print("pretraining ended {0} - it {1} - lval {2}".format(i, j, lval));
                    break;
            

            
    def build_complete_net(self, input):
        """
        Builds a complete network of all encoding layers 

        Parameters
        ----------
        self : Autoencoder
        input : Tensor
            Input placeholder used to feed data to the network

        Returns
        -------
        list
            List of Tensor objects that create the network

        """
        net = [];
        inp = input;
        for i in range(0, len(self.weights)):
            inp = self.create_layer(i, inp);
            net.append(inp);
            
        return net;

    
    def build_pretrain_net(self, n, input):
        """
        Builds a partially frozen and uncomplete network used for pretraining steps

        Parameters
        ----------
        self : Autoencoder
        n : int
            Layer index
        input : Tensor
            Input placeholder used to feed data to the network

        Returns
        -------
        list
            List of Tensor objects that create the network
            
        """
        layers = [];
        inp = input;
        for i in range(0, n):
            inp = self.create_layer(i, inp, is_fixed = True);
            layers.append(inp);
        
        inp = self.create_layer(n, inp);
        layers.append(inp);

        inp = self.create_layer(n, inp, is_decoder = True);
        layers.append(inp);
        
        for i in range(0, n):
            inp = self.create_layer(n - 1 - i, inp, is_fixed = True, is_decoder = True);
            layers.append(inp);
        return layers;




class Classifier(Model):
    """
    Class used to append a classifier to a pretrained autoencoder

    """

    def __init__(self, autoencoder, outputs):
        """
        Class constructor

        Parameters
        ----------
        self : Classifier 
        autoencoder : Autoencoder
            The autoencoder object to which the classifier will be appended to

        """
        self.autoencoder = autoencoder;
        self.input_placeholder = tf.placeholder("float", [None, self.autoencoder.input_count]);
        self.encoder = autoencoder.build_complete_net(self.input_placeholder);
        input = self.encoder[len(self.encoder) - 1];
        self.weights = tf.Variable(tf.random_normal([input.shape[1].value, outputs]));
        self.biases = tf.Variable(tf.random_normal([outputs]));
        self.layer = tf.nn.softmax(tf.matmul(input, self.weights) + self.biases);
        self.output_placeholder = tf.placeholder("float", [None, outputs]);
        self.get_accuracy_tensors();
        
    def create_train_summary(self, data, output, test_data, test_output, train_suits, test_suits):
        def add_values_for_whole_set(s, set_name, results):
            s.value.add(tag="{0} acc - exact".format(set_name), simple_value=results[0])
            s.value.add(tag="{0} acc - off by 1".format(set_name), simple_value=results[1])
            s.value.add(tag="{0} acc - off by 2".format(set_name), simple_value=results[2])

        def add_suit_values_for_set(s, set_name, results, suits):
            initial = 0;
            if suits != 4:
                add_values_for_whole_set(s, "No trump - {0}".format(set_name), results[0]);
                initial = 1;

            if suits != 1:
                add_values_for_whole_set(s, "Spades - {0}".format(set_name), results[initial]);
                add_values_for_whole_set(s, "Hearts - {0}".format(set_name), results[initial + 1]);
                add_values_for_whole_set(s, "Diamonds - {0}".format(set_name), results[initial + 2]);
                add_values_for_whole_set(s, "Clubs - {0}".format(set_name), results[initial + 3]);

        s = tf.Summary();
        r_tr = self.test(data, output);
        r_test = self.test(test_data, test_output);
        suits_tr = self.suit_based_accurancy(data, output, train_suits);
        suits_test = self.suit_based_accurancy(test_data, test_output, test_suits);
        # save accuracy for whole sets
        add_values_for_whole_set(s, "Train", r_tr);
        add_values_for_whole_set(s, "Test", r_test);
        # save accuracy for suits
        add_suit_values_for_set(s, "Train", suits_tr, train_suits);
        add_suit_values_for_set(s, "Test", suits_test, test_suits);
        return s;

    def train(self, data, desired_output, learning_rate, it, delta, path, train_data, train_output, test_data, test_output, train_suits = 5, test_suits = 5, loss_f = Model.mse_loss, no_improvement = 5, experiment_name = ""):
        """
        Main train method
    
        This method is used to start the fine tuning phase of the classifier and the autoencoder layers

        Parameters
        ----------
        self : Classifier 
        data : list
            List of numpy arrays with training inputs which will be fed to the input placeholder
        desired_output : list
            List of numpy arrays with labels for the corresponding training inputs
        learning_rate : float
            Learning rate for the optimizer used
        it : int
            Iterations count to be executed
        delta : float
            If the improvement between epochs is smaller than delta, the training process is aborted
        path : string
            Path used to store summaries generated by Tensorflow
        test_data : list 
            Test inputs used for validation
        test_output : list
            Test outputs used for validation
        loss_f : Tensor
            Loss function used by the optimizer
        no_improvement : int
            If the error function value worsens the amount of times specified by this parameter the calculation will be aborted
            
        """
        loss = loss_f(self.output_placeholder, self.layer); 
        opt = tf.train.RMSPropOptimizer(learning_rate);
        optimizer = opt.minimize(loss);
        self.autoencoder.session.run(tf.variables_initializer([self.weights, self.biases]));
        slot_vars = [self.weights, self.biases] + self.autoencoder.biases + self.autoencoder.weights;
        self.autoencoder.session.run(Model.initialize_optimizer(opt, slot_vars));
        hist_summaries = [(self.autoencoder.weights[i], 'weights{0}'.format(i)) for i in range(0, len(self.autoencoder.weights))];
        hist_summaries.extend([(self.autoencoder.biases[i], 'biases{0}'.format(i)) for i in range(0, len(self.autoencoder.weights))]);
        summaries = [tf.summary.histogram(v[1], v[0]) for v in hist_summaries];
        summaries.append(tf.summary.scalar("loss_4", loss));   
        summary_op = tf.summary.merge(summaries);

        writer = tf.summary.FileWriter(path, graph=self.autoencoder.session.graph)

        if delta > 0:
            prev_val = 0;
            current_val = 0;
            no_improvement_counter = 0;
            it_counter = 0;
            while True:
                for k in range(0, len(data)):
                    lval, _, summary = self.autoencoder.session.run([loss, optimizer, summary_op], feed_dict={self.input_placeholder: data[k], self.output_placeholder: desired_output[k]});
                if it_counter % 100 == 0:
                    s = self.create_train_summary(train_data, train_output, test_data, test_output, train_suits, test_suits);
                    current_val = self.test(test_data, test_output)[0];
                    print(current_val);
                    self.save_model(experiment_name + " at {0}".format(it_counter))
                    print("finetuning - it {0} - lval {1}".format(it_counter, lval));
                    writer.add_summary(summary, it_counter);
                    writer.add_summary(s, it_counter);
                    if prev_val != 0 and (current_val - prev_val) < delta:
                        print(current_val - prev_val);
                        if(no_improvement_counter > no_improvement):
                            print("terminating due to no improvement");
                            print("finetuning - it {0} - lval {1}".format(it_counter, lval));
                            return
                        else:
                            no_improvement_counter = no_improvement_counter + 1;
                    prev_val = current_val;
                it_counter = it_counter + 1;

        else:
            for i in range(0, it):
                for k in range(0, len(data)):
                    lval, _, summary = self.autoencoder.session.run([loss, optimizer, summary_op], feed_dict={self.input_placeholder: data[k], self.output_placeholder: desired_output[k]});
                if i % 100 == 0:
                    print("finetuning - it {0} - lval {1}".format(i, lval));
                    writer.add_summary(summary, i);

    def get_accuracy_tensors(self):
        """
        Method used to set up tensors holding accuracy values

        Parameters
        ----------
        self : Classifier 
        data : list
            List of numpy arrays with training inputs which will be fed to the input placeholder
        desired_output : list
            List of numpy arrays with labels for the corresponding training inputs
            
        Returns
        -------
        (Tensor, Tensor, Tensor)
            Tuple containing all tensors hold accuracy values
        """
        correct_prediction = tf.equal(tf.argmax(self.layer, 1), tf.argmax(self.output_placeholder, 1));
        missed_by_one = tf.less_equal(tf.abs(tf.argmax(self.layer, 1) - tf.argmax(self.output_placeholder, 1)), 1);
        missed_by_two = tf.less_equal(tf.abs(tf.argmax(self.layer, 1) - tf.argmax(self.output_placeholder, 1)), 2);

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self.accuracy_missed_by_one = tf.reduce_mean(tf.cast(missed_by_one, "float"))
        self.accuracy_missed_by_two = tf.reduce_mean(tf.cast(missed_by_two, "float"))
        return (self.accuracy, self.accuracy_missed_by_one, self.accuracy_missed_by_two);
        
    def test(self, data, desired_output):
        """
        Test method
    
        This method is used to obtain the accurancy of the created model including information such as:
            - Exact deal match percentage
            - Actual deal missed by 1 percentage
            - Actual deal missed by 2 percentage

        Parameters
        ----------
        self : Classifier 
        data : list
            List of numpy arrays with training inputs which will be fed to the input placeholder
        desired_output : list
            List of numpy arrays with labels for the corresponding training inputs
            
        Returns
        -------
        Detailed accurancy concerning the current model
        """
        return self.autoencoder.session.run([self.accuracy, self.accuracy_missed_by_one, self.accuracy_missed_by_two], feed_dict={self.input_placeholder: data, self.output_placeholder: desired_output});

    def suit_based_accurancy(self, test_data, test_labels, suits):
        """
        Advanced testing method
    
        This method is used to obtain the accurancy of the created model including information such as:
            - Exact deal match percentage
            - Actual deal missed by 1 percentage
            - Actual deal missed by 2 percentage

        This method is the extended version of the test method, as it calculates the above mentioned values separately
        for every suit i.e. separately for No Trump, Diamonds Trump, Hearts Trump, Spades Trump and Clubs Trump games. 

        Parameters
        ----------
        self : Classifier 
        test_data : list
            List of numpy arrays with training inputs which will be fed to the input placeholder
        test_labels : list
            List of numpy arrays with labels for the corresponding training inputs
        suits : int
            A number indicating the number of suits in the input data    

        Returns
        -------
        list
            List of results for each suit
        """
        l = len(test_data);
        res = [];
        for i in range(0, suits):        
            input = [test_data[x] for x in range(0, l) if x % (suits * 4) in range(4 * i, 4*i + 4)];
            labels = [test_labels[x] for x in range(0, l) if x % (suits * 4) in range(4 * i, 4*i + 4)];
            res.append(self.test(input, labels));
        return res;

    def save_model(self, name):
        """
        Save method

        This method stores the current model (including the autoencoder it contains) on permament memory 
        under the location ./name .

        Parameters
        ----------
        self : Classifier 
        name : string
            Model filename
        test_labels : list
            List of numpy arrays with labels for the corresponding training inputs
        suits : int
            A number indicating the number of suits in the input data    
        """
        saver = tf.train.Saver();
        saver.save(self.autoencoder.session, "./models/{0}".format(name));

    def restore_model(self, name):
        """
        Restore method

        This method restores a before saved model (including the autoencoder it contains) from permament memory. 
        Please remember to setup as instance of the Classifier and Autoencoder classes before running it, as well
        as starting a Tensorflow session.

        Parameters
        ----------
        self : Classifier 
        name : string
            Model filename
        """
        saver = tf.train.Saver();
        saver.restore(self.autoencoder.session, "./models/{0}".format(name));

    def multi_batch_test(self, suits, data_batches, outputs_batches, batch_count):
        """
        Multibatch testing method

        This method aggregates the results for several batches.

        Parameters
        ----------
        self : Classifier 
        suits : int
            Number of suits
        data_batches : list
            List of input data batches
        outputs_batches : list
            List of output data batches
        batch_count : int
            Number of batches
        """
        res = [];
        for i in range(0, batch_count):
            res.append(self.suit_based_accurancy(data_batches[i], outputs_batches[i], suits));

        return ([sum(z)/batch_count for x in list(zip(*res)) for z in zip(*x)], res);
