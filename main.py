import pickle
import numpy as np
np.random.seed(1337) # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.regularizers import activity_l2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import keras.utils.layer_utils as layer_utils

pickle_file = open("IIed.p", "rb")
X_train = pickle.load(pickle_file)
pickle_file.close()
X_train = X_train.reshape(50000,1000,1)
X_train = X_train.astype("float32")
print(X_train.shape[0],X_train[0].shape[0], 'train samples')

# # # Layer-wise pre-training
trained_encoders = []

X_train_tmp = X_train
nb_hidden_layer = 3
nb_filter=30
filter_length=21
batch_size = 1
nb_epoch = 5

for n in range(nb_hidden_layer):
    print('Pre-training the layer: {}'.format(n))
    # Create AE and training
    encoder = Sequential()
    encoder.add(Convolution1D(input_length=1000,
                            nb_filter = nb_filter,
                            input_dim = 1,
                            filter_length = filter_length,
                            border_mode = "same",
                            activation = "tanh",
                            activity_regularizer=activity_l2(0.01),
                            subsample_length = 1))
    decoder = Sequential()
    decoder.add(Convolution1D(input_length=1000,
                            input_dim = nb_filter,
                            nb_filter = 1,
                            filter_length = filter_length,
                            border_mode = "same",
                            activation = "tanh",
                            subsample_length = 1))
    ae=Sequential()
    ae.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True))
    layer_utils.print_layer_shapes(ae,[(1,1000,1)]) 
    print "....compile"
    ae.compile(loss='mean_squared_error', optimizer='rmsprop')
    print "....fitting"
    ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch)
    break;
#     # Store trainined weight
#     # trained_encoders.append(ae.layers[0].encoder)
#     # Update training data
#     # X_train_tmp = ae.predict(X_train_tmp)
#     break;

# # # Fine-tuning
# # print('Fine-tuning')
# # model = Sequential()
# # for encoder in trained_encoders:
# #     model.add(encoder)
# # model.add(Dense(nb_hidden_layers[-1], nb_classes, activation='softmax'))

# # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# # model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
# #           show_accuracy=True, validation_data=(X_test, Y_test))
# # score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
# # print('Test score:', score[0])
# # print('Test accuracy:', score[1])









# # # from sklearn.cross_validation import KFold

# # # from keras.preprocessing import sequence
# # # from keras.models import Sequential
# # # from keras.layers.core import Dense, Dropout, Activation, Flatten
# # # from keras.layers.embeddings import Embedding
# # # from keras.layers.convolutional import Convolution1D, MaxPooling1D

# # # # set parameters:
# # # nb_filter = 250
# # # filter_length = 21
# # # batch_size = 32
# # # hidden_dims = 500
# # # nb_epoch = 5
# # # seq_len = 1050

# # # X_train_tmp = X_train
# # # model = Sequential()
# # # model.add(Convolution1D(nb_filter = nb_filter,
# # #                         filter_length = filter_length,
# # #                         border_mode = "valid",
# # #                         activation = "tanh",
# # #                         subsample_length = 1))
# # # # model.add(MaxPooling1D(pool_length = 2))
# # # model.add(Convolution1D(nb_filter = nb_filter,
# # #                         filter_length = filter_length,
# # #                         border_mode = "valid",
# # #                         activation = "linear",
# # #                         subsample_length = 1))
# # # model.compile(loss = "mean_squared_error", optimizer = "rmsprop")

# # # # print("Pad sequences (samples x time)")
# # # # X_train = sequence.pad_sequences(X_train, maxlen = seq_len)
# # # # X_test = sequence.pad_sequences(X_test, maxlen = seq_len)
# # # # print('X_train shape:', X_train_tmp.shape)

# # # # model.fit(X_train_tmp, X_train_tmp, batch_size = batch_size,
# # # #                                     nb_epoch = nb_epoch,
# # # #                                     show_accuracy = True,
# # # #                                     validation_data = (X_test, y_test))











