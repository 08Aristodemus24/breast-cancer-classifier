# -*- coding: utf-8 -*-
"""breast_cancer_train_nn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k3aEvjzTYjE2yFrKdLVilldB0PxRvx30

# Breast Cancer Wisconsin Binary Classifier
## Import libraries
"""

import tensorflow as tf
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy as bce_loss
from tensorflow.keras.metrics import BinaryAccuracy, BinaryCrossentropy as bce_metric
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform, RandomNormal, RandomUniform, HeNormal, HeUniform
from tensorflow.keras.optimizers import Adadelta, Adafactor, Adagrad, Adam, AdamW, Adamax, Ftrl, Nadam, RMSprop, SGD 

import json

def load_baseline():
    # define model architecture
    model = Sequential([
        Dense(units=10, activation='relu',),
        Dense(units=10, activation='relu',),
        Dense(units=10, activation='relu',),
        Dense(units=10, activation='relu',),
        Dense(units=1, activation='linear'),
    ])

    model.compile(
        loss=bce_loss(from_logits=True),
        optimizer=Adam(learning_rate=0.001),
        metrics=[bce_metric(from_logits=True), BinaryAccuracy(threshold=0.5)]
    )

    return model

def load_tuned(param_file_path: str):
    # define model architecture using results from search_best_params notebook
    with open(param_file_path) as in_file:
        hp_str = in_file.read()
        best_hyper_params = json.loads(hp_str)
        in_file.close()

    print(best_hyper_params)

    alt_hyper_params = {
        'learning_rate': 0.0075,
        'lambda': 0.9,
        'activation': 'relu',
        'optimizer': 'Adam'
    }
    
    initializers = {
        'GlorotNormal': GlorotNormal(),
        'GlorotUniform': GlorotUniform(),
        'RandomNormal': RandomNormal(mean=0.0, stddev=1.0),
        'RandomUniform': RandomUniform(minval=-0.05, maxval=0.05),
        'HeNormal': HeNormal(),
        'HeUniform': HeUniform()
    }

    optimizers = {
        'Adadelta': Adadelta(learning_rate=alt_hyper_params['learning_rate']),
        'Adafactor': Adafactor(learning_rate=alt_hyper_params['learning_rate']),
        'Adagrad': Adagrad(learning_rate=alt_hyper_params['learning_rate']),
        'Adam': Adam(learning_rate=alt_hyper_params['learning_rate']),
        'AdamW': AdamW(learning_rate=alt_hyper_params['learning_rate']),
        'Adamax': Adamax(learning_rate=alt_hyper_params['learning_rate']), 
        'Ftrl': Ftrl(learning_rate=alt_hyper_params['learning_rate']),
        'Nadam': Nadam(learning_rate=alt_hyper_params['learning_rate']),
        'RMSprop': RMSprop(learning_rate=alt_hyper_params['learning_rate']),
        'SGD': SGD(learning_rate=alt_hyper_params['learning_rate'])
    }

    # update this such that number of layers are not static but dynamic
    model = Sequential()

    # number of hidden layers
    for l in range(best_hyper_params['layer_num']):
        print(f'building layer {l + 1}')
        
        # number of nodes per layer
        model.add(Dense(
            units=best_hyper_params[f'layer_{l + 1}'], 
            activation=alt_hyper_params['activation'], 
            kernel_initializer=initializers[best_hyper_params['initializer']],
            kernel_regularizer=L2(alt_hyper_params['lambda'])))
        
        model.add(Dropout(best_hyper_params['dropout']))

    model.add(Dense(units=1, activation='linear', kernel_regularizer=L2(alt_hyper_params['lambda'])))
    
    model.compile(
        optimizer=optimizers[alt_hyper_params['optimizer']],
        loss=bce_loss(from_logits=True),
        metrics=[bce_metric(), BinaryAccuracy(threshold=0.5)]
    )

    return model