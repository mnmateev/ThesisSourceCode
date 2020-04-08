import sys
import mido
import numpy as np
import pandas as pd
from random import *
from mido import MidiFile, MidiTrack, Message
from tensorflow.python.keras import backend as K
from keras import objectives, losses
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.layers import Input, Dense, Activation, LSTM, Dropout, Flatten, RepeatVector, Lambda
from tensorflow.python.keras.models import Sequential, Model, load_model, model_from_json
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import predictMethods
import zLayer

# network parameters
epochs = sys.argv[1]
batch_size = sys.argv[2]
original_dim = 8 * 96 * 96
input_shape = (original_dim,)
intermediate_dim = 200
latent_dim = 120

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def initModel():
    print('creating model')

    original_dim = 8 * 96 * 96

    # network parameters
    input_shape = (original_dim,)
    intermediate_dim = 200
    batch_size = 16
    latent_dim = 120

    x = Input(shape=input_shape)
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean, name='8_vae_mlp')

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    reconstruction_loss = losses.binary_crossentropy(x, x_decoded_mean)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # return model
    return vae

def compileModel(vae):
    vae.compile(optimizer='adam')
    return vae


def trainModel(model, X, epochs, batch_size):
    model.fit(X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2)

#  Load Dataset

print("Loading Data...")
xSamples = np.load('Samples/volume10.npy')
numSamples = xSamples.shape[0]
print("Loaded " + str(numSamples) + " samples.")

# we have loaded samples - individual measures
# turn into 16 measure data points:

n = int(numSamples / 8)
sampleSongs = np.ndarray(shape=(n,8,96,96), dtype = np.int32)

for i in range(n):
    tempSong = np.ndarray(shape=(8, 96, 96), dtype = np.int32)
    for j in range(8):
        tempSong[j] = xSamples[i+j]
    sampleSongs[i] = tempSong

X = sampleSongs
#X = X[700:1350]
modelSamples = X.shape[0]

print('samples reshaped into ' + str(modelSamples) + ' samples for model')

# initialize model
print('loading model')
vae = load_model('vae')
#vae, generator = initModel()
vae = compileModel(vae)

newX = np.ndarray(shape=(len(X),73728), dtype = np.int32)
for i in range(len(X)):
    sample = X[i]
    sample = sample.flatten()
    sample = sample.flatten()
    newX[i] = sample

print(newX.shape)

'''
trainModel(vae, newX, int(epochs), int(batch_size))


print('saving model')
vae.save('vae', save_format="tf")
'''

#'''
# make prediction
print('making prediction')
newX = newX.astype('float32')
Xsong = vae.predict(newX)
print('saving prediction')

newXsong = np.ndarray(shape=(len(Xsong),8,96,96), dtype = np.float32)
for i in range(len(Xsong)):
    sample = Xsong[i]
    sample = sample.reshape(8,96,96)
    newXsong[i] = sample

Xsong = newXsong

opus = "Opus10/op10m2"

predictMethods.samples_to_midi(Xsong[0]+Xsong[1], opus+'n1', 16)
predictMethods.samples_to_midi(Xsong[4]+Xsong[5], opus+'n2', 16)
predictMethods.samples_to_midi(Xsong[8]+Xsong[9], opus+'n3', 16)
predictMethods.samples_to_midi(Xsong[11]+Xsong[12], opus+'n4', 16)
predictMethods.samples_to_midi(Xsong[16]+Xsong[17], opus+'n5', 16)
predictMethods.samples_to_midi(Xsong[25]+Xsong[26], opus+'n6', 16)
predictMethods.samples_to_midi(Xsong[28]+Xsong[29], opus+'n7', 16)
predictMethods.samples_to_midi(Xsong[32]+Xsong[33], opus+'n8', 16)
#'''
