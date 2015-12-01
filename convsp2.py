from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import theano
import sys
np.random.seed(1337)  # for reproducibility
import pickle
from keras.models import Sequential
from keras.layers import containers
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.regularizers import activity_l2, activity_l1, l1, l2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, AutoEncoder
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding2D
import keras.utils.layer_utils as layer_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from seya.layers.coding import ConvSparseCoding

floatX = theano.config.floatX
theano.config.optimizer='None'

# -- file in
if len(sys.argv) == 1:
	print("what to run ??")
	sys.exit();
filename = sys.argv[1]

# -- construct pickle
print('pickling '+filename)
sys.stdout.flush();
fo = open("data/"+filename+".csv", "rb")
lines = fo.readlines();
total = [];
for line in lines:
    l = line.split("\n")[0].split(",")[0:-1];
    l = [float(i) for i in l];
    total.append(l);
    # print(total);

X_train = np.array(total);
# pickle.dump( total, open( "data/pickle/"+filename+".p", "wb" ) )
# sys.exit();

# -- load data
# pickle_file = open("data/pickle/"+filename+".p", "rb")
# X_train = np.array(pickle.load(pickle_file))
# pickle_file.close()
X_train = X_train.reshape(X_train.shape[0],1,1,1000)
# X_train = X_train[0:500]
X_train = X_train.astype(floatX)
print(X_train.shape, 'train samples')

# -- modeling sparse coding theano
model = Sequential()
conv = ConvSparseCoding(nb_filter=1, stack_size=40, nb_row=1, nb_col=31,
                 input_row=1, input_col=1000,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 W_regularizer=l2(0.01),
                 activity_regularizer=None,
                 return_reconstruction=True, n_steps=10, truncate_gradient=-1,
                 gamma=0.1)
model.add(conv)

# -- optimize
adg = Adagrad(lr=0.01, epsilon=1e-6)
model.compile(loss='mean_squared_error', optimizer=adg)
model.fit(X_train, X_train, batch_size=10, nb_epoch=1)

# -- get reconstruction for comparison
X_recon = model.predict(X_train)

# -- get code for building Encoder
conv.return_reconstruction=False;
model.compile(loss='mean_squared_error', optimizer=adg)
Z_predict = model.predict(X_train)

# -- save results
pickle.dump( Z_predict, open( "result/z_"+filename+".p", "wb" ) )
pickle.dump( X_recon, open( "result/recon_"+filename+".p", "wb" ) )





