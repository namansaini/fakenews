# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:07:05 2019

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import queue

import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.python.compiler import eager as tfe
#import tensorflow.contrib.eager as tfe
from main import num_validation_samples,num_labeled_samples,num_train_unlabeled_samples,NUM_TEST_SAMPLES,NUM_TRAIN_SAMPLES
from main import label_data,val_data,unlabel_data,test_data,labels
# Enable Eager Execution
#tf.enable_eager_execution()
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


from main import TempEnsemModel, temporal_ensembling_gradients, ramp_up_function, ramp_down_function
import main


def mains():

    # Editable variables

    batch_size = 100
    epochs = 30
    max_learning_rate = 0.0002 # 0.001 as recomended in the paper leads to unstable training. 
    initial_beta1 = 0.9
    final_beta1 = 0.5
    alpha = 0.6
    max_unsupervised_weight = 30 * num_labeled_samples /\
                               (NUM_TRAIN_SAMPLES - num_validation_samples)
    checkpoint_directory = './checkpoints/TemporalEnsemblingModel'
    tensorboard_logs_directory = './logs/TemporalEnsemblingModel'

    # Assign it as tfe.variable since we will change it across epochs
    learning_rate = tf.Variable(max_learning_rate)
    beta_1 = tf.Variable(initial_beta1)


    # You can replace it by the real ratio (preferably with a big batch size : num_labeled_samples / num_train_unlabeled_samples
    # This means that the labeled batch size will be labeled_batch_fraction * batch_size and the unlabeled batch size will be
    # (1-labeled_batch_fraction) * batch_size
    labeled_batch_fraction = num_labeled_samples / num_train_unlabeled_samples
    batches_per_epoch = round(
        num_labeled_samples/(batch_size * labeled_batch_fraction))

    batches_per_epoch_val = int(round(num_validation_samples / batch_size))

    model = TempEnsemModel()
    # Paper has beta2=0.990 but I experimented decreasing it a little bit (as recomended in the paper) and it led
    # to more stable training
    optimizer = tf.optimizers.Adam(
        learning_rate=learning_rate, beta_1=beta_1, beta_2=0.980)

    best_val_accuracy = 0
    global_step = tf.compat.v1.train.get_or_create_global_step()
    writer = tf.summary.create_file_writer(tensorboard_logs_directory)
    writer.set_as_default()

    # Ensemble predictions - the first samples of the array are for the labeled samples
    # and the remaining ones are for the unlabeled samples.
    # The Z and z are the notation used in the paper
    Z = np.zeros((NUM_TRAIN_SAMPLES, 10))
    z = np.zeros((NUM_TRAIN_SAMPLES, 10))
    # variable needed if you use a batch ratio different than the true ratio
    sample_epoch = np.zeros((NUM_TRAIN_SAMPLES, 1))

    for epoch in range(epochs):
        rampdown_value = ramp_down_function(epoch, epochs)
        # In the paper the authors use 80 as the epoch with max rampup_value
        rampup_value = ramp_up_function(epoch, 40)

        if epoch == 0:
            unsupervised_weight = 0
        else:
            unsupervised_weight = max_unsupervised_weight * \
                                  rampup_value

        learning_rate.assign(rampup_value * rampdown_value * max_learning_rate)
        beta_1.assign(rampdown_value * initial_beta1 +
                      (1.0 - rampdown_value) * final_beta1)

        epoch_loss_avg = tf.metrics.Mean()
        epoch_accuracy = tf.metrics.Accuracy()
        epoch_loss_avg_val = tf.metrics.Mean()
        epoch_accuracy_val = tf.metrics.Accuracy()

        for batch_nr in range(batches_per_epoch):

            labeled=label_data.sample(batch_size)
            X_unlabeled_train=unlabel_data.sample(batch_size)

            X_labeled_train=labeled.drop(columns=['type'])
            labeled_indexes=labeled.index.values
            y_labeled_train = to_categorical(np.asarray(labels[labeled.index]))
            unlabeled_indexes=X_unlabeled_train.index.values

            #X_labeled_train, y_labeled_train, labeled_indexes = train_labeled_iterator.get_next()
            #X_unlabeled_train, _, unlabeled_indexes = train_unlabeled_iterator.get_next()

            # We need to correct labeled samples indexes (in Z the first num_labeled_samples samples are for ensemble labeled predictions)
            current_ensemble_indexes = np.concatenate(
                [labeled_indexes, unlabeled_indexes + num_labeled_samples])
            current_ensemble_targets = z[current_ensemble_indexes]

            current_outputs, loss_val, grads = temporal_ensembling_gradients(X_labeled_train, y_labeled_train, X_unlabeled_train,
                                                                             model, unsupervised_weight, current_ensemble_targets)

            optimizer.apply_gradients(zip(grads, model.variables),
                                      global_step=global_step)

            epoch_loss_avg(loss_val)
            epoch_accuracy(tf.argmax(model(X_labeled_train), 1),
                           tf.argmax(y_labeled_train, 1))

            epoch_loss_avg(loss_val)
            epoch_accuracy(
                tf.argmax(model(X_labeled_train), 1), tf.argmax(y_labeled_train, 1))

            Z[current_ensemble_indexes, :] = alpha * \
                Z[current_ensemble_indexes, :] + (1-alpha) * current_outputs
            z[current_ensemble_indexes, :] = Z[current_ensemble_indexes, :] * \
                (1. / (1. - alpha **
                       (sample_epoch[current_ensemble_indexes] + 1)))
            sample_epoch[current_ensemble_indexes] += 1

            if (batch_nr == batches_per_epoch - 1):
                for batch_val_nr in range(batches_per_epoch_val):
                    val= val_data.sample(batch_size)
                    X_val=val.drop(columns=['type'])
                    y_val=to_categorical(np.asarray(labels[val.index]))
                    #X_val, y_val, _ = validation_iterator.get_next()
                    y_val_predictions = model(X_val, training=False)

                    epoch_loss_avg_val(tf.losses.softmax_cross_entropy(
                        y_val, y_val_predictions))
                    epoch_accuracy_val(
                        tf.argmax(y_val_predictions, 1), tf.argmax(y_val, 1))

        print("Epoch {:03d}/{:03d}: Train Loss: {:9.7f}, Train Accuracy: {:02.6%}, Validation Loss: {:9.7f}, "
              "Validation Accuracy: {:02.6%}, lr={:.9f}, unsupervised weight={:5.3f}, beta1={:.9f}".format(epoch+1,
                                                                                                           epochs,
                                                                                                           epoch_loss_avg.result(),
                                                                                                           epoch_accuracy.result(),
                                                                                                           epoch_loss_avg_val.result(),
                                                                                                           epoch_accuracy_val.result(),
                                                                                                           learning_rate.numpy(),
                                                                                                           unsupervised_weight,
                                                                                                           beta_1.numpy()))

        # If the accuracy of validation improves save a checkpoint
        if best_val_accuracy < epoch_accuracy_val.result():
            best_val_accuracy = epoch_accuracy_val.result()
            checkpoint = tf.Checkpoint(optimizer=optimizer,
                                        model=model,
                                        optimizer_step=global_step)
            checkpoint.save(file_prefix=checkpoint_directory)

        # Record summaries
        with tf.summary.record_summaries_every_n_global_steps(1):
            tf.summary.scalar('Train Loss', epoch_loss_avg.result())
            tf.summary.scalar(
                'Train Accuracy', epoch_accuracy.result())
            tf.summary.scalar(
                'Validation Loss', epoch_loss_avg_val.result())
            tf.summary.histogram(
                'Z', tf.convert_to_tensor(Z), step=global_step)
            tf.summary.histogram(
                'z', tf.convert_to_tensor(z), step=global_step)
            tf.summary.scalar(
                'Validation Accuracy', epoch_accuracy_val.result())
            tf.summary.scalar(
                'Unsupervised Weight', unsupervised_weight)
            tf.summary.scalar('Learning Rate', learning_rate.numpy())
            tf.summary.scalar('Ramp Up Function', rampup_value)
            tf.summary.scalar('Ramp Down Function', rampdown_value)

    print('\nTrain Ended! Best Validation accuracy = {}\n'.format(best_val_accuracy))

    # Load the best model
    root = tf.Checkpoint(optimizer=optimizer,
                          model=model,
                          optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Evaluate on the final test set
    num_test_batches = math.ceil(NUM_TEST_SAMPLES/batch_size)
    test_accuracy = tf.metrics.Accuracy()
    for test_batch in range(num_test_batches):
        test = test_data.sample(batch_size)
        X_test = test.drop(columns=['type'])
        y_test = to_categorical(np.asarray(labels[test.index]))
        #X_test, y_test, _ = test_iterator.get_next()
        y_test_predictions = model(X_test, training=False)
        test_accuracy(tf.argmax(y_test_predictions, 1), tf.argmax(y_test, 1))

    print("Final Test Accuracy: {:.6%}".format(test_accuracy.result()))


if __name__ == "__main__":
    mains()