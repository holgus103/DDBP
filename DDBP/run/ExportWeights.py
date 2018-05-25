import tensorflow as tf
import time
import random;
import numpy;
import sys;
import matplotlib.pyplot as plt;

sys.path.append("./../");

import models;
import data_parser as dp
#"no_trump_altered_104enc_eta=0.002 at 24000"

l = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'];

def plot(data, path, ticks, tick_labels):
    f = plt.figure(figsize=(2000/96, 1500/96), dpi=96)
    fix, ax = plt.subplots();

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(5)
    plt.imshow(data, cmap="hsv");
    plt.colorbar()
    plt.savefig(path, dpi=400)
    plt.close(f)

def restore_and_plot_altered(first_model, second_model, autoencoder_count, classifier_counts):

    # create models
    a = models.Autoencoder(models.Model.cross_entropy_loss, autoencoder_count);
    c = models.Classifier(a, classifier_counts);
    c.restore_model(first_model);
    # get initial weights
    w_initial = a.session.run(a.weights[0])
    # load second model
    c.restore_model(second_model);
    # get final weights
    w_final = a.session.run(a.weights[0])
    ticks = [x+0.5 for x in range(0, 52)];
    tick_labels =  l+l+l+l


    for i in range(0,4):
        plot(w_initial[i], "initial{0}".format(i), ticks, tick_labels)

    for i in range(0,4):
        plot(w_final[i], "final{0}".format(i), ticks, tick_labels)
    # print(w_final[0] - w_initial[0])
    for i in range(0,4):
        plot(w_final[i] - w_initial[i], "diff{0}".format(i), ticks, tick_labels)



def restore_and_plot_simple(first_model, second_model, autoencoder_counts, classifier_counts):

    # create models
    a = models.Autoencoder(models.Model.cross_entropy_loss, autoencoder_counts);
    c = models.Classifier(a, classifier_counts);
    c.restore_model(first_model);
    # get initial weights
    w_initial = a.session.run(a.weights[0])
    # load second model
    c.restore_model(second_model);
    # get final weights
    w_final = a.session.run(a.weights[0])

    ticks = [x+0.5 for x in range(0, 208)];
    labels = l + l + l + l
    tick_labels =  labels + labels + labels + labels

    plot(w_initial, "initial", ticks, tick_labels)

    plot(w_final, "final", ticks, tick_labels)
    # print(w_final[0] - w_initial[0])

    plot(w_final - w_initial, "diff", ticks, tick_labels)


# restore_and_plot_altered("no_trump_altered_156enc_eta=0.002_2 at 0",\
# "no_trump_altered_156enc_eta=0.002_2 at 42000",\
# 39,\
# [52,13,14])

restore_and_plot_simple()


#"trump_altered_156enc_eta=0.004_2 at 24000"
