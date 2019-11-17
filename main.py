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


def embedding():
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
    return vocabulary_size,embedding_matrix

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
        vocabulary_size, embedding_matrix=embedding()
        self.embedding_layer_title = Embedding(vocabulary_size,EMBEDDING_DIM, weights=[embedding_matrix])
        self.reshape_title = Reshape((sequence_length_title, EMBEDDING_DIM, 1))
        self.conv_0_title = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM), activation='relu',
                              kernel_regularizer=regularizers.l2(0.01))
        self.conv_1_title = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM), activation='relu',
                              kernel_regularizer=regularizers.l2(0.01))
        self.conv_2_title = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM), activation='relu',
                                   kernel_regularizer=regularizers.l2(0.01))
        self.maxpool_0_title = MaxPooling2D((sequence_length_title - filter_sizes[0] + 1, 1), strides=(1, 1))
        self.maxpool_1_title = MaxPooling2D((sequence_length_title - filter_sizes[1] + 1, 1), strides=(1, 1))
        self.maxpool_2_title = MaxPooling2D((sequence_length_title - filter_sizes[2] + 1, 1), strides=(1, 1))
        self.dense_title = Dense(50, activation='relu', kernel_regularizer='l2', name='DenseTitle')
        self.conv_0_text = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM), activation='relu',
                             kernel_regularizer=regularizers.l2(0.01))
        self.conv_1_text = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM), activation='relu',
                             kernel_regularizer=regularizers.l2(0.01))
        self.conv_2_text = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM), activation='relu',
                             kernel_regularizer=regularizers.l2(0.01))

        self.maxpool_0_text = MaxPooling2D((sequence_length_text - filter_sizes[0] + 1, 1), strides=(1, 1))
        self.maxpool_1_text = MaxPooling2D((sequence_length_text - filter_sizes[1] + 1, 1), strides=(1, 1))
        self.maxpool_2_text = MaxPooling2D((sequence_length_text - filter_sizes[2] + 1, 1), strides=(1, 1))
        self. dense_text = Dense(100, activation='relu', kernel_regularizer='l2', name='DenseText')
        self.dense1 = Dense(50, activation='relu')
        self.dropout1= Dropout(drop)
        self.dense2 = Dense(50, activation='relu')
        self.dropout2= Dropout(drop)
        self.out = Dense(2, activation='softmax')



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


    def call(self, input, training=True):
        """ Function that allows running a tensor through the pi model
        Arguments:2
            input {[tensor]} -- batch of images
            training {bool} -- if true applies augmentaton and additive noise
        Returns:
            [tensor] -- predictions
        """
        
        title = input.title
        text = input.text
        sequences_train_text = tokenizer.texts_to_sequences(text)
        sequences_train_title = tokenizer.texts_to_sequences(title)

        title = pad_sequences(sequences_train_text)
        text = pad_sequences(sequences_train_title)
        
        #title layer
        #inputs_title = Input(shape=(sequence_length_title,))
        h1 = self.embedding_layer(title,training)
        h1 = self.reshape_title(h1,training)
        #reshape_title = Reshape((sequence_length_title,EMBEDDING_DIM,1))(embedding_title)
        if training:
            h1 = self.__aditive_gaussian_noise(h1, 0.15)
        conv_layer0_title = self.conv_0_title(h1,training)
        conv_layer1_title = self.conv_1_title(h1,training)
        conv_layer2_title = self.conv_2_title(h1,training)
        
        maxpool_layer0_title = self.maxpool_0_title(conv_layer0_title)
        maxpool_layer1_title = self.maxpool_1_title(conv_layer1_title)
        maxpool_layer2_title = self.maxpool_2_title(conv_layer2_title)

        merged_tensor_title = concatenate([maxpool_layer0_title, maxpool_layer1_title, maxpool_layer2_title], axis=1)

        flattenTitle = Flatten()(merged_tensor_title)
        reshapeTitle = Reshape((3*num_filters,))(flattenTitle)
        denseTitle = self.dense_title(reshapeTitle,training)
        #dense_title = Dense(50, activation='relu', kernel_regularizer='l2', name='DenseTitle')(reshape)
        
        #text layer
        #inputs_text = Input(shape=(sequence_length_text,))
        h2 = self.embedding_layer(text)
        h2 = self.reshape_title(h2,training)
        
        if training:
            h2 = self.__aditive_gaussian_noise(h2, 0.15)
        conv_layer0_text = self.conv_0_title(h2,training)
        conv_layer1_text = self.conv_1_title(h2,training)
        conv_layer2_text = self.conv_2_title(h2,training)
        
        maxpool_layer0_text = self.maxpool_0_title(conv_layer0_text)
        maxpool_layer1_text = self.maxpool_1_title(conv_layer1_text)
        maxpool_layer2_text = self.maxpool_2_title(conv_layer2_text)
        
        '''
        conv_0_text = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_text)
        conv_1_text = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_text)
        conv_2_text = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_text)
        
        maxpool_0_text = MaxPooling2D((sequence_length_text - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0_text)
        maxpool_1_text = MaxPooling2D((sequence_length_text - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1_text)
        maxpool_2_text = MaxPooling2D((sequence_length_text - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2_text) '''
        
        merged_tensor_text = concatenate([maxpool_layer0_text, maxpool_layer1_text, maxpool_layer2_text], axis=1)
        flattenText = Flatten()(merged_tensor_text)
        reshapeText = Reshape((3*num_filters,))(flattenText)
        denseText = self.dense_text(reshapeText)
        
        x = concatenate([denseTitle, denseText])
        
        #Common part
        x = self.dense1(x)
        x = self.droput1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.out(x)
'''  x = Dense(50, activation='relu')(x)
        x = Dropout(drop)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(drop)(x)
        out = Dense(4, activation='softmax')(x)
        return out '''


''' conv_0_title = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_title)
        conv_1_title = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_title)
        conv_2_title = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_title)

        maxpool_0_title = MaxPooling2D((sequence_length_title - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0_title)
        maxpool_1_title = MaxPooling2D((sequence_length_title - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1_title)
        maxpool_2_title = MaxPooling2D((sequence_length_title - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2_title) '''







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

