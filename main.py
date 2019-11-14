import numpy as np
import pandas as pd
import tensorflow as tf
data=pd.read_csv("fake.csv", usecols=[2,4,5,8,19])
test_data=data.sample(frac=0.3,random_state=200)
train_data=data.drop(test_data.index)

train_data=train_data.dropna()
test_data=test_data.dropna()

authors=train_data.author.unique()
dic={}
for i,auth in enumerate(authors):
    dic[auth]=i
train_data.author=train_data.author.apply(lambda x:dic[x])
sites=train_data.site_url.unique()
dic={}
for i,site in enumerate(sites):
    dic[site]=i
train_data.site_url=train_data.site_url.apply(lambda x:dic[x])

types=train_data.type.unique()
dic={}
for i,type in enumerate(types):
    dic[type]=i
labels=train_data.type.apply(lambda x:dic[x])

val_data=train_data.sample(frac=0.2,random_state=200)
train_data=train_data.drop(val_data.index)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

texts=train_data.text
titles=train_data.title

NUM_WORDS=20000
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)
tokenizer.fit_on_texts(texts)
sequences_train_text = tokenizer.texts_to_sequences(texts)
sequences_valid_text=tokenizer.texts_to_sequences(val_data.text)
tokenizer.fit_on_texts(titles)
sequences_train_title = tokenizer.texts_to_sequences(titles)
sequences_valid_title=tokenizer.texts_to_sequences(val_data.title)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train_text = pad_sequences(sequences_train_text)
X_val_text = pad_sequences(sequences_valid_text,maxlen=X_train_text.shape[1])
y_train = to_categorical(np.asarray(labels[train_data.index]))
y_val = to_categorical(np.asarray(labels[val_data.index]))
print('Shape of X train and X validation tensor:', X_train_text.shape,X_val_text.shape)
print('Shape of label train and validation tensor:', y_train.shape,y_val.shape)


X_train_title = pad_sequences(sequences_train_title)
X_val_title = pad_sequences(sequences_valid_title,maxlen=X_train_title.shape[1])

print('Shape of X train and X validation tensor:', X_train_title.shape,X_val_title.shape)

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

EMBEDDING_DIM=300
vocabulary_size=min(len(word_index)+1,NUM_WORDS)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

del(word_vectors)

from tensorflow.keras.layers import Embedding
embedding_layer = Embedding(vocabulary_size,EMBEDDING_DIM, weights=[embedding_matrix],trainable=True)

from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from tensorflow.keras.layers.core import Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers




class TempEnsemModel(tf.keras.Model):
    """ Class for defining eager compatible tfrecords file
        I did not use tfe.Network since it will be depracated in the
        future by tensorflow.
    """

    def __init__(self):
        """ Init
            Set all the layers that need to be tracked in the process of
            gradients descent (pooling and dropout for example dont need
            to be stored)
        """

        super(TempEnsemModel, self).__init__()
        

    def __aditive_gaussian_noise(self, input, std):
        """ Function to add additive zero mean noise as described in the paper
        Arguments:
            input {tensor} -- image
            std {int} -- std to use in the random_normal
        Returns:
            {tensor} -- image with added noise
        """

        noise = tf.random_normal(shape=tf.shape(
            input), mean=0.0, stddev=std, dtype=tf.float32)
        return input + noise


    def call(self, X_train_title,X_train_text, training=True):
        """ Function that allows running a tensor through the pi model
        Arguments:
            input {[tensor]} -- batch of images
            training {bool} -- if true applies augmentaton and additive noise
        Returns:
            [tensor] -- predictions
        """

        
            
        
        sequence_length_title = X_train_title.shape[1]
        sequence_length_text=X_train_text.shape[1]
        filter_sizes = [2,3,4]
        num_filters = 100
        drop = 0.2
        
        
        #title layer
        inputs_title = Input(shape=(sequence_length_title,))
        embedding_title = embedding_layer(inputs_title)
        reshape_title = Reshape((sequence_length_title,EMBEDDING_DIM,1))(embedding_title)
        if training:
            reshape_title = self.__aditive_gaussian_noise(reshape_title, 0.15)
        
        conv_0_title = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_title)
        conv_1_title = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_title)
        conv_2_title = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_title)
        
        maxpool_0_title = MaxPooling2D((sequence_length_title - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0_title)
        maxpool_1_title = MaxPooling2D((sequence_length_title - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1_title)
        maxpool_2_title = MaxPooling2D((sequence_length_title - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2_title)
        
        merged_tensor_title = concatenate([maxpool_0_title, maxpool_1_title, maxpool_2_title], axis=1)
        flatten = Flatten()(merged_tensor_title)
        reshape = Reshape((3*num_filters,))(flatten)
        dense_title = Dense(50, activation='relu', kernel_regularizer='l2', name='DenseTitle')(reshape)
        
        #text layer
        inputs_text = Input(shape=(sequence_length_text,))
        embedding_text = embedding_layer(inputs_text)
        reshape_text = Reshape((sequence_length_text,EMBEDDING_DIM,1))(embedding_text)
        
        if training:
            reshape_text = self.__aditive_gaussian_noise(reshape_text, 0.15)
            
        conv_0_text = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_text)
        conv_1_text = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_text)
        conv_2_text = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_text)
        
        maxpool_0_text = MaxPooling2D((sequence_length_text - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0_text)
        maxpool_1_text = MaxPooling2D((sequence_length_text - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1_text)
        maxpool_2_text = MaxPooling2D((sequence_length_text - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2_text)
        
        merged_tensor_text = concatenate([maxpool_0_text, maxpool_1_text, maxpool_2_text], axis=1)
        flatten = Flatten()(merged_tensor_text)
        reshape = Reshape((3*num_filters,))(flatten)
        dense_text = Dense(100, activation='relu', kernel_regularizer='l2', name='DenseText')(reshape)
        
        x = concatenate([dense_title, dense_text])
        
        #Common part
        x = Dense(50, activation='relu')(x)
        x = Dropout(drop)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(drop)(x)
        out = Dense(4, activation='softmax')(x)
        return out








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


# this creates a model that includes
model = Model(inputs, output)

adam = Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])
callbacks = [EarlyStopping(monitor='val_loss')]
model.fit(X_train, y_train, batch_size=1000, epochs=10, verbose=1, validation_data=(X_val, y_val),
         callbacks=callbacks)  # starts training
sequences_test=tokenizer.texts_to_sequences(test_data.text)
X_test = pad_sequences(sequences_test,maxlen=X_train.shape[1])
y_pred=model.predict(X_test)