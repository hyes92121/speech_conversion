#####################################
#   import general use modeules     #
#####################################
import os
import sys
import time
import random
from random import randint
import argparse
import numpy as np
import matplotlib.pyplot as plt

#####################################
#  import librosa related modeules  #
#####################################
import librosa
import librosa.display
import spectrogram as spg
from hyperparams import Hyperparams as hp

#####################################
#   import keras related modeules   #
#####################################
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM
from keras.layers import Input, Dense, Flatten, Activation, Reshape, concatenate
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.callbacks import ModelCheckpoint

#####################################
#       load data from csv          #
#####################################
def load_data(filename):
    label, time_series = [], []
    data = np.load(filename)
    for l, t in data:
        label.append(l)
        time_series.append(t)
    return np.array(label), np.array(time_series)

###########################################
#           data preprocessing            #
#  data format: [(one-hot, time series)]  #
###########################################
def preprocess(wav, padding=False, stft=False):
    """
    unzip = lambda x: zip(*x)
    label, wav  = unzip(data)           # returns as tuples
    """
    #label, wav =  # converts into numpy arrays
    if padding:
        print('padding sequences')
        s = time.time()
        max_len = len(max(wav, key=len))
        wav = sequence.pad_sequences(wav, maxlen=max_len, padding='post', dtype='float32')
        e = time.time()
        print("used time: {}".format(e-s))
    if stft:
        print('stft...')
        s = time.time()
        wav = np.array([librosa.stft(x) for x in wav])
        np.save('stft.npy', wav)
        e = time.time()
        print("used time: {}".format(e-s))

    return wav, max_len

###########################################
#    get mag and phase from spectrogram   #
###########################################
def get_mag_phase(D):
    mag, phase = [], []
    for spec in D:
        m, p = librosa.magphase(spec)
        mag.append(m)
        phase.append(p)
    return np.array(mag), np.array(phase)

    

#####################################
#           build encoder           #
#####################################
def build_encoder(max_len):
    spectral_frame   = Input(shape=(max_len, )) 
    speaker_identity = Input(shape=(10, ))

    encoded = Dense(256, activation='relu')(spectral_frame)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(64 , activation='relu')(encoded)

    latent_with_speaker_id = concatenate([encoded, speaker_identity])

    encoder = Model([spectral_frame, speaker_identity], latent_with_speaker_id)
    return encoder 


#####################################
#           build decoder           #
#####################################
def build_decoder(encoder_output_length):
    decoder_input = Input(shape=(encoder_output_length, ))
    
    decoded     = Dense(256, activation='relu')(decoder_input)
    decoded     = Dense(256, activation='relu')(decoded)
    decoded     = Dense(1025*498, activation='sigmoid')(decoded) 

    decoder     = Model(decoder_input, decoded)
    return decoder 


"""
#####################################
#           build deep NN           #
#####################################
def build_model(input_length):
    input_layer = Input(shape(input_length, ))
    encoded     = Dense(256, activation='relu')(input_layer)
    encoded     = Dense(128, activation='relu')(encoded)
    encoded     = Dense(64 , activation='relu')(encoded)

    decoded     = Dense(128, activation='relu')(encoded)
    decoded     = Dense(256, activation='relu')(decoded)
    decoded     = Dense(input_length, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')  

    return autoencoder
"""

#####################################
#             train NN              #
#####################################
def train(x, y, model, epochs=100, batch_size=256, validation=0.1):
    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation)
    return model



if __name__ == "__main__":
    l, t = load_data("data/train_label.npy")
    print(l)
    wav_ft, max_len = preprocess(t, padding=True, stft=False)
    wav_ft = np.load('stft.npy')
    """
    s = time.time()
    mag, phase = get_mag_phase(wav_ft)
    e = time.time()
    print('magphase: {}'.format(e-s))
    np.save('mag.npy', mag)
    """
    mag = np.load('mag.npy')

    mag = np.reshape(mag, (1620, 1025*498))

    ##################################################################
    #                      building autoencoder                      #
    ##################################################################
    spectral_frame      = Input(shape=(1025*498, ))
    speaker_identity    = Input(shape=(10, ))

    encoded             = Dense(256, activation='relu')(spectral_frame)
    encoded             = Dense(256, activation='relu')(encoded)
    encoded             = Dense(64 , activation='relu')(encoded)

    latent_wtth_spkerid = concatenate([encoded, speaker_identity])

    decoded             = Dense(256, activation='relu')(latent_wtth_spkerid)
    decoded             = Dense(256, activation='relu')(decoded)
    decoded             = Dense(1025*498, activation=None)(decoded)

    autoencoder         = Model([spectral_frame, speaker_identity], decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    ##################################################################
    """
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    """
    autoencoder.fit([mag, l], mag, validation_split=0.05, epochs=20, batch_size=128, verbose=1)
    #autoencoder = train([mag, l], mag, autoencoder)
    autoencoder.save('autoencoder.h5')

    test_data = mag[:3]
    test_label = label[:3]
    #test_label = [label[234], label[974], label[1105]]
    
    predict = autoencoder.predict([test_data, test_label], batch_size=128)

    predict = np.reshape(predict, (3, 1025, 498))

    D = predict[0]*phase[0]
    y = librosa.istft(D)
    librosa.output.write_wav('decode.wav', y, 22050)
