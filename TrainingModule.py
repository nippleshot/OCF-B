import warnings
warnings.filterwarnings('ignore')
import keras
import numpy
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.constraints import non_neg
from keras.layers import concatenate
from keras.layers import dot
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
os.chdir("/content/drive/MyDrive/GraduationLab/Graduation/GraduationCode")

def lossView(history):
    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = numpy.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def zeroInjectionMF(dataset, epo, NlatentFactor):
  with tf.device('/device:GPU:0') :
    dataset.user_id = dataset.user_id.astype('category').cat.codes.values
    dataset.item_id = dataset.item_id.astype('category').cat.codes.values
    train = dataset

    n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
    n_latent_factors = NlatentFactor

    # Model build :
    movie_input = keras.layers.Input(shape=[1],name='Item')
    movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

    user_input = keras.layers.Input(shape=[1],name='User')
    user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)

    prod = keras.layers.dot([movie_vec, user_vec], axes=1, normalize=False, name='DotProduct')

    model = Model(inputs=[user_input, movie_input], outputs=prod)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Model fit :
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    codeDirectory = os.path.dirname(os.getcwd())
    mfModelPath = os.path.join(codeDirectory, "GraduationCode", "MFModel", "best_model.h5")
    model_checkpoint = ModelCheckpoint(filepath=mfModelPath, monitor='val_loss', save_best_only=True)
    history = model.fit([train.user_id, train.item_id], train.rating, validation_split=0.33, epochs=epo, verbose=1, callbacks=[early_stopping, model_checkpoint])
    lossView(history)

    return history, train

def trainMFModel(dataset, epo, NlatentFactor, testSize):
  with tf.device('/device:GPU:0') :
    dataset.user_id = dataset.user_id.astype('category').cat.codes.values
    dataset.item_id = dataset.item_id.astype('category').cat.codes.values
    train, test = train_test_split(dataset, test_size=testSize)

    n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
    n_latent_factors = NlatentFactor

    # Model build :
    movie_input = keras.layers.Input(shape=[1],name='Item')
    movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

    user_input = keras.layers.Input(shape=[1],name='User')
    user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)

    prod = keras.layers.dot([movie_vec, user_vec], axes=1, normalize=False, name='DotProduct')

    model = Model(inputs=[user_input, movie_input], outputs=prod)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Model fit :
    history = model.fit([train.user_id, train.item_id], train.rating, validation_split=0.33, epochs=epo, verbose=1)
    lossView(history)

    return history, model, test, train






