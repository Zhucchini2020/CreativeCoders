import os
import keras
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model
from keras.optimizers.legacy import RMSprop
from keras.models import Model,Sequential
from hyperparams import *

def autoencoder_code(create_new_encoder: bool, input_img, decoder, encoder, train_X, train_ground, valid_X, valid_ground):
    "trains/saves or loads autoencoder weights depending on create_new_weights bool, returning autoencoder"
    autoencoder = Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
    if create_new_encoder:
        autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=BATCH_SIZE,epochs=AUTOENCODER_EPOCHS,verbose=1,validation_data=(valid_X, valid_ground))
        loss = autoencoder_train.history['loss']
        val_loss = autoencoder_train.history['val_loss']
        epochs = range(AUTOENCODER_EPOCHS)
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        autoencoder.save_weights('autoencoder.h5')
    else:
        autoencoder.load_weights('autoencoder.h5')
    return autoencoder


