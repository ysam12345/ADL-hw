import numpy as np
import h5py

from glob import glob

import tensorflow as tf
from tensorflow.keras import layers

import json
from gensim.models.word2vec import Word2Vec

from tqdm import tqdm

from dataset import load_dataset_from_json, prepare_dataset
#from nltk.tokenize import word_tokenize


HDF5_FILE_PATH = './dataset.hdf5'
DATA_NUM = None

if HDF5_FILE_PATH not in glob('./*'):
    print("### Load data from json file ###")
    dataset = load_dataset_from_json()
    h5f = h5py.File(HDF5_FILE_PATH, 'w')
    X_train, Y_train = prepare_dataset(dataset['train'], DATA_NUM)
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('Y_train', data=Y_train)
    X_val, Y_val = prepare_dataset(dataset['valid'], DATA_NUM)
    h5f.create_dataset('X_val', data=X_val)
    h5f.create_dataset('Y_val', data=Y_val)
    X_test, Y_test = prepare_dataset(dataset['test'], DATA_NUM)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()
else:
    # load data from HDF5 file
    print("### Load data from hd5f file ###")
    h5f = h5py.File(HDF5_FILE_PATH,'r')
    X_train = h5f['X_train'][:]
    Y_train = h5f['Y_train'][:]
    X_val = h5f['X_val'][:]
    Y_val = h5f['Y_val'][:]
    X_test = h5f['X_test'][:]
    Y_test = h5f['Y_test'][:]

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=100, output_dim=64))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))
    model.summary()
    return model

print("@@@@@@@@@@@@@@@@")
print(X_train.shape)
print(Y_train.shape)
print(sum([i for i in list(Y_train)]))
print("@@@@@@@@@@@@@@@@")
model = build_model()
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=128 * 64,
          epochs=500,
          verbose=1,
          validation_data=(X_val, Y_val))