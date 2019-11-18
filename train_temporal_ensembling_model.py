import numpy as np
import pandas as pd
import math
import pandas as pd
import tensorflow as tf


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from tensorflow.keras.layers import Embedding

data=pd.read_csv("all_data.csv", usecols=['author','main_img_url','site_url','text','title','type'], nrows = 200)

data=data.dropna()

test_data=data.sample(frac=0.3,random_state=20)
train_data=data.drop(test_data.index)
NUM_TRAIN_SAMPLES = train_data.shape[0]
NUM_TEST_SAMPLES = test_data.shape[0]

def encode_label(column):
    columns = column.unique()
    dic = {}
    for i, col in enumerate(columns):
        dic[col] = i
    return column.apply(lambda x: dic[x])


train_data.author=encode_label(train_data.author)

train_data.site_url=encode_label(train_data.site_url)

labels=encode_label(train_data.type)



val_data=train_data.sample(frac=0.2,random_state=20)
train_data=train_data.drop(val_data.index)

unlabel_data = train_data.sample(frac=0.5,random_state=20)

label_data = train_data.drop(unlabel_data.index)

unlabel_data = unlabel_data.drop(columns=['type'])

test_data = test_data.reset_index(drop=True)
label_data= label_data.reset_index(drop=True)
unlabel_data = unlabel_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

num_labeled_samples =label_data.shape[0]
num_validation_samples = val_data.shape[0]
num_train_unlabeled_samples =unlabel_data.shape[0]


EMBEDDING_DIM = 300
NUM_WORDS=20000
tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)


def create_embedding_layer():
    tokenizer.fit_on_texts(train_data.text)
    tokenizer.fit_on_texts(val_data.text)

    tokenizer.fit_on_texts(train_data.title)
    tokenizer.fit_on_texts(val_data.title)

    word_index = tokenizer.word_index
    word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

    del (word_vectors)
    embedding = Embedding(vocabulary_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)
    return embedding

    # print('Found %s unique tokens.' % len(word_index))





#print('Shape of X train and X validation tensor:', X_train_text.shape,X_val_text.shape)
#print('Shape of label train and validation tensor:', y_train.shape,y_val.shape)

sequences_train_text = tokenizer.texts_to_sequences(train_data.text)
#sequences_valid_text = tokenizer.texts_to_sequences(val_data.text)
sequences_train_title = tokenizer.texts_to_sequences(train_data.title)
#sequences_valid_title = tokenizer.texts_to_sequences(val_data.title)


X_train_text = pad_sequences(sequences_train_text)
#X_val_text = pad_sequences(sequences_valid_text, maxlen=X_train_text.shape[1])
X_train_title = pad_sequences(sequences_train_title)
#X_val_title = pad_sequences(sequences_valid_title,maxlen=X_train_title.shape[1])

#y_train = to_categorical(np.asarray(labels[train_data.index]))
#y_val = to_categorical(np.asarray(labels[val_data.index]))


sequence_length_title = X_train_title.shape[1]
sequence_length_text=X_train_text.shape[1]
del (X_train_text,X_train_title,sequences_train_text,sequences_train_title)

filter_sizes = [2, 3, 4]
num_filters = 100
drop = 0.2

# prepare embedding layer


from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate,Reshape,Flatten
#from tensorflow.keras.layers.core import Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


embedding_layer=create_embedding_layer()

inputs_title = Input(shape=(sequence_length_title,))
embedding_title = embedding_layer(inputs_title)
reshape_title = Reshape((sequence_length_title, EMBEDDING_DIM, 1))(embedding_title)


conv_0_title = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM), activation='relu',
                      kernel_regularizer=regularizers.l2(0.01))(reshape_title)
conv_1_title = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM), activation='relu',
                      kernel_regularizer=regularizers.l2(0.01))(reshape_title)
conv_2_title = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM), activation='relu',
                      kernel_regularizer=regularizers.l2(0.01))(reshape_title)

maxpool_0_title = MaxPooling2D((sequence_length_title - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0_title)
maxpool_1_title = MaxPooling2D((sequence_length_title - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1_title)
maxpool_2_title = MaxPooling2D((sequence_length_title - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2_title)

merged_tensor_title = concatenate([maxpool_0_title, maxpool_1_title, maxpool_2_title], axis=1)
flatten = Flatten()(merged_tensor_title)
reshape = Reshape((3 * num_filters,))(flatten)
dense_title = Dense(50, activation='relu', kernel_regularizer='l2', name='DenseTitle')(reshape)

# text layer
inputs_text = Input(shape=(sequence_length_text,))
embedding_text = embedding_layer(inputs_text)
reshape_text = Reshape((sequence_length_text, EMBEDDING_DIM, 1))(embedding_text)



conv_0_text = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM), activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(reshape_text)
conv_1_text = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM), activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(reshape_text)
conv_2_text = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM), activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(reshape_text)

maxpool_0_text = MaxPooling2D((sequence_length_text - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0_text)
maxpool_1_text = MaxPooling2D((sequence_length_text - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1_text)
maxpool_2_text = MaxPooling2D((sequence_length_text - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2_text)

merged_tensor_text = concatenate([maxpool_0_text, maxpool_1_text, maxpool_2_text], axis=1)
flatten = Flatten()(merged_tensor_text)
reshape = Reshape((3 * num_filters,))(flatten)
dense_text = Dense(100, activation='relu', kernel_regularizer='l2', name='DenseText')(reshape)

x = concatenate([dense_title, dense_text])

# Common part
x = Dense(50, activation='relu')(x)
x = Dropout(drop)(x)
x = Dense(50, activation='relu')(x)
x = Dropout(drop)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=[inputs_title,inputs_text], outputs=output)




def temporal_ensembling_loss(X_train_labeled, y_train_labeled, X_train_unlabeled, model, unsupervised_weight, ensembling_targets):
    """ Gets the loss for the temporal ensembling model
    Arguments:
        X_train_labeled {tensor} -- labeled samples
        y_train_labeled {tensor} -- labeled train labels
        X_train_unlabeled {tensor} -- unlabeled samples 
        model {tf.keras.Model} -- temporal ensembling model
        unsupervised_weight {float} -- weight of the unsupervised loss
        ensembling_targets {np.array} --  ensembling targets
    Returns:
        {tensor} -- predictions for the ensembles
        {tensor} -- loss value
    """

    z_labeled = model(X_train_labeled) #X_train_labled_title,X_train_labeled_text
    z_unlabeled = model(X_train_unlabeled) #X_train_unlabled_title,X_train_unlabeled_text

    current_predictions = tf.concat([z_labeled, z_unlabeled], 0)

    return current_predictions, tf.losses.softmax_cross_entropy(
        y_train_labeled, z_labeled) + unsupervised_weight * (
            tf.losses.mean_squared_error(ensembling_targets, current_predictions))


def temporal_ensembling_gradients(X_train_labeled, y_train_labeled, X_train_unlabeled, model, unsupervised_weight, ensembling_targets):
    """ Gets the gradients for the temporal ensembling model
    Arguments:
        X_train_labeled {tensor} -- labeled samples
        y_train_labeled {tensor} -- labeled train labels
        X_train_unlabeled {tensor} -- unlabeled samples 
        model {tf.keras.Model} -- temporal ensembling model
        unsupervised_weight {float} -- weight of the unsupervised loss
        ensembling_targets {np.array} --  ensembling targets
    Returns:
        {tensor} -- predictions for the ensembles
        {tensor} -- loss value
        {tensor} -- gradients for each model variables
    """

    with tf.GradientTape() as tape:
        ensemble_precitions, loss_value = temporal_ensembling_loss(X_train_labeled, y_train_labeled, X_train_unlabeled,
                                                                   model, unsupervised_weight, ensembling_targets)

    return ensemble_precitions, loss_value, tape.gradient(loss_value, model.variables)



def ramp_up_function(epoch, epoch_with_max_rampup=80):
    """ Ramps the value of the weight and learning rate according to the epoch
        according to the paper
    Arguments:
        {int} epoch
        {int} epoch where the rampup function gets its maximum value
    Returns:
        {float} -- rampup value
    """

    if epoch < epoch_with_max_rampup:
        p = max(0.0, float(epoch)) / float(epoch_with_max_rampup)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0


def ramp_down_function(epoch, num_epochs):
    """ Ramps down the value of the learning rate and adam's beta
        in the last 50 epochs according to the paper
    Arguments:
        {int} current epoch
        {int} total epochs to train
    Returns:
        {float} -- rampup value
    """
    epoch_with_max_rampdown = 50

    if epoch >= (num_epochs - epoch_with_max_rampdown):
        ep = (epoch - (num_epochs - epoch_with_max_rampdown)) * 0.5
        return math.exp(-(ep * ep) / epoch_with_max_rampdown)
    else:
        return 1.0

# train_temporal starts here after merging  the files


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

# Enable Eager Execution
#tf.enable_eager_execution()
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical





def mains():

    # Editable variables

    batch_size = 10
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
    Z = np.zeros((NUM_TRAIN_SAMPLES, 2))
    z = np.zeros((NUM_TRAIN_SAMPLES, 2))
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

            labeled=label_data.sample(n=batch_size)
            X_unlabeled_train=unlabel_data.sample(n=batch_size)
            
            y_labeled_train=labeled.type.apply(lambda x: 0 if (x =='fake') else 1).values
            X_labeled_train=labeled.drop(columns=['type'])
            labeled_indexes=labeled.index.values
            
            y_labeled_train = to_categorical(y_labeled_train,num_classes=2)
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
                    val= val_data.sample(n=batch_size)
                    y_val = val.type.apply(lambda x: 0 if (x =='fake') else 1).values
                    X_val=val.drop(columns=['type'])
                    y_val=to_categorical(y_val,num_classes=2)
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
                          optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Evaluate on the final test set
    num_test_batches = math.ceil(NUM_TEST_SAMPLES/batch_size)
    test_accuracy = tf.metrics.Accuracy()
    for test_batch in range(num_test_batches):
        test = test_data.sample(n=batch_size)
        y_test = test.type.apply(lambda x: 0 if (x =='fake') else 1).values
        X_test = test.drop(columns=['type'])
        y_test = to_categorical(y_test,num_classes=2)
        #X_test, y_test, _ = test_iterator.get_next()
        y_test_predictions = model(X_test, training=False)
        test_accuracy(tf.argmax(y_test_predictions, 1), tf.argmax(y_test, 1))

    print("Final Test Accuracy: {:.6%}".format(test_accuracy.result()))


if __name__ == "__main__":
    mains()