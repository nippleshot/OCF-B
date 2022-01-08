import warnings
warnings.filterwarnings('ignore')
import keras
import numpy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import dot
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=5) # 조기종료 콜백함수 정의

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
    user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors, name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)

    prod = keras.layers.dot([movie_vec, user_vec], axes=1, normalize=False, name='DotProduct')
    prod = keras.activations.sigmoid(prod)

    model = Model(inputs=[user_input, movie_input], outputs=prod)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Model fit :
    history = model.fit([train.user_id, train.item_id], train.rating, validation_split=0.33, epochs=epo, verbose=0)
    lossView(history)

    return history, model, test, train

def trainMLPModel(dataset, epo, NlatentUser, NlatentItem, testSize):
  with tf.device('/device:GPU:0') :
    dataset.user_id = dataset.user_id.astype('category').cat.codes.values
    dataset.item_id = dataset.item_id.astype('category').cat.codes.values
    train, test = train_test_split(dataset, test_size=testSize)

    n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())

    n_latent_factors_user = NlatentUser
    n_latent_factors_movie = NlatentItem

    movie_input = keras.layers.Input(shape=[1],name='Item')
    movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

    user_input = keras.layers.Input(shape=[1],name='User')
    user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input))

    concat = concatenate([movie_vec, user_vec], name='Concat')
    concat_dropout = keras.layers.Dropout(0.2)(concat)

    dense_1 = keras.layers.Dense(64, activation='relu', name='FullyConnected-1')(concat_dropout)
    dense_batch_1 = keras.layers.BatchNormalization(name='Batch-1')(dense_1)
    dropout_1 = keras.layers.Dropout(0.2, name='Dropout-1')(dense_batch_1)

    dense_2 = keras.layers.Dense(32, activation='relu', name='FullyConnected-2')(dropout_1)
    dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)
    dropout_2 = keras.layers.Dropout(0.2, name='Dropout-2')(dense_batch_2)

    dense_3 = keras.layers.Dense(16, activation='relu', name='FullyConnected-3')(dropout_2)
    dense_4 = keras.layers.Dense(8, activation='relu', name='FullyConnected-4')(dense_3)

    result = keras.layers.Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='Prediction')(dense_4)
    adam = Adam(lr=0.01)
    model = Model([user_input, movie_input], result)
    model.compile(optimizer=adam, loss= 'binary_crossentropy')

    history = model.fit([train.user_id, train.item_id], train.rating, validation_split=0.33, batch_size=256, epochs=epo, verbose=0)
    lossView(history)

    return history, model, test, train


def trainNeuMFModel(dataset, epo, NlatentUser, NlatentItem, nlatentMF, testSize):
  with tf.device('/device:GPU:0') :
    dataset.user_id = dataset.user_id.astype('category').cat.codes.values
    dataset.item_id = dataset.item_id.astype('category').cat.codes.values
    train, test = train_test_split(dataset, test_size = testSize)

    n_latent_factors_user = NlatentUser
    n_latent_factors_movie = NlatentItem
    n_latent_factors_mf = nlatentMF
    n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())

    movie_input = keras.layers.Input(shape=[1],name='Item')
    movie_embedding_mlp = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding-MLP')(movie_input)
    movie_vec_mlp = keras.layers.Flatten(name='FlattenMovies-MLP')(movie_embedding_mlp)
    movie_embedding_mf = keras.layers.Embedding(n_movies + 1, n_latent_factors_mf, name='Movie-Embedding-MF')(movie_input)
    movie_vec_mf = keras.layers.Flatten(name='FlattenMovies-MF')(movie_embedding_mf)

    user_input = keras.layers.Input(shape=[1],name='User')
    user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding-MLP')(user_input))
    user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(keras.layers.Embedding(n_users + 1, n_latent_factors_mf,name='User-Embedding-MF')(user_input))

    concat = concatenate([movie_vec_mlp, user_vec_mlp], name='Concat')
    dropout_0 = keras.layers.Dropout(0.2, name='Dropout-0')(concat)
    dense_1 = keras.layers.Dense(64, activation='relu' ,name='FullyConnected-1')(dropout_0)
    dense_batch_1 = keras.layers.BatchNormalization(name='Batch-1')(dense_1)
    dropout_1 = keras.layers.Dropout(0.2, name='Dropout-1')(dense_batch_1)
    dense_2 = keras.layers.Dense(32, activation='relu' ,name='FullyConnected-2')(dropout_1)
    dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)
    dropout_2 = keras.layers.Dropout(0.2, name='Dropout-2')(dense_batch_2)
    dense_3 = keras.layers.Dense(16, activation='relu', name='FullyConnected-3')(dropout_2)

    pred_mlp = keras.layers.Dense(8, activation='relu', name='FullyConnected-4')(dense_3)
    pred_mf = dot([movie_vec_mf, user_vec_mf], axes=1, name='Dot')
    combine_mlp_mf = concatenate([pred_mf, pred_mlp], name='Concat-MF-MLP')

    result = keras.layers.Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='Prediction')(combine_mlp_mf)

    model = Model([user_input, movie_input], result)
    adam = Adam(lr=0.01)
    model.compile(optimizer=adam, loss= 'binary_crossentropy')

    history = model.fit([train.user_id, train.item_id], train.rating, validation_split=0.33, batch_size=256, epochs=epo, verbose=0)
    lossView(history)

    return history, model, test, train
