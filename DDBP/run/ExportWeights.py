import tensorflow as tf
import time
import random;
import numpy;
import sys;
import matplotlib.pyplot as plt;

sys.path.append("./../");

import models;
import data_parser as dp


def restore_and_lot(model_name, autoencoder_count, classifier_counts, weights_index):
    a = models.Autoencoder(models.Model.cross_entropy_loss, autoencoder_count);
    c = models.Classifier(a, classifier_counts);
    c.restore_model(model_name);

    l = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'];
    labels = l + l + l + l; 

    fix, ax = plt.subplots();
    w = a.session.run(a.weights[0]);
    ticks = [x+0.5 for x in range(0, 52)];
    ax.set_yticks(ticks)
    ax.set_yticklabels(l + l + l + l)
    plt.imshow(w[weights_index]);
    plt.colorbar()
    plt.show()


#"trump_altered_156enc_eta=0.004_2 at 24000"