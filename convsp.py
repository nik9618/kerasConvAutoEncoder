from keras.layers.convolutional import ConvolutionalSparseCoding
import os
import numpy as np
import theano
import matplotlib.pyplot as plt
import pickle
np.random.seed(1337) # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.regularizers import activity_l2, activity_l1, l1, l2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding2D
import keras.utils.layer_utils as layer_utils
from keras.optimizers import SGD, Adam, RMSprop
floatX = theano.config.floatX
theano.config.optimizer='None'

# load data
pickle_file = open("IIed.p", "rb")
X_train = pickle.load(pickle_file)
pickle_file.close()
X_train = X_train.reshape(50000,1,1,1000)
X_train = X_train[0:4000]
X_train = X_train.astype(floatX)
print(X_train.shape, 'train samples')

# modelling
nb_filter=50
input_dim = 1
filter_length=31
batch_size = 1
nb_epoch = 1

model = Sequential()
model.add(
    ConvolutionalSparseCoding(nb_filter=1, stack_size=nb_filter, nb_row=1, nb_col=filter_length,
                 input_row=1, input_col=1000,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 W_regularizer=None,activity_regularizer=None,
                 return_reconstruction=True, n_steps=10, truncate_gradient=-1,
                 gamma=0.0001)
)
rmsp = RMSprop(lr=.001)
print "compiling"
model.compile(loss='mse', optimizer=rmsp)

# layer_utils.print_layer_shapes(model,[(1,1,1,1000)]) 

# optimization
print "fitting"
model.fit(X_train, # input 
          X_train, # and output are the same thing, since we are doing generative modeling.
          batch_size=batch_size,
          nb_epoch=nb_epoch)
X_train_tmp = model.predict(X_train)






















