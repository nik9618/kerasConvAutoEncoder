from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import pickle
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb

from seya.layers.coding import ConvSparseCoding
import os
import numpy as np
import theano
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
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
floatX = theano.config.floatX
theano.config.optimizer='None'



pickle_file = open("IIed.p", "rb")
X_train = pickle.load(pickle_file)
pickle_file.close()
X_train = X_train.reshape(50000,1,1,1000)
X_train = X_train[0:30000]
X_train = X_train.astype(floatX)
print(X_train.shape, 'train samples')

# -- model
model = Sequential()
model.add(
    ConvSparseCoding(nb_filter=1, stack_size=30, nb_row=1, nb_col=31,
                 input_row=1, input_col=1000,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 W_regularizer=None,
                 activity_regularizer=None,
                 return_reconstruction=True, n_steps=10, truncate_gradient=-1,
                 gamma=0.005)
)

adg = Adagrad(lr=0.01, epsilon=1e-6)
model.compile(loss='mean_squared_error', optimizer=adg)
layer_utils.print_layer_shapes(model,[(1,1,1,1000)])

model.fit(X_train, X_train, batch_size=10, nb_epoch=1)
X_train_tmp = model.predict(X_train)





