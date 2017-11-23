import Autoencoder;
import tensorflow as tf;

a = Autoencoder.Autoencoder(10, [8,2], tf.Variable(tf.random_normal([1,  10])));
