# -*- coding: utf-8 -*-
"""breast_cancer_train_nn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k3aEvjzTYjE2yFrKdLVilldB0PxRvx30

# Breast Cancer Wisconsin Binary Classifier
## Import libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utilities.data_preprocessor import preprocess
from utilities.data_visualizer import view_train_cross
from models.baseline_model_arc import load_baseline

import json

"""### check current working directory"""

import os
print(os.getcwd())

"""## loading the data"""
# use path below if in local machine
df = pd.read_csv('./data.csv')

# use path below if in google collab
# df = pd.read_csv('./sample_data/breast_cancer_data.csv')

X, Y = preprocess(df)
X_trains, X_, Y_trains, Y_ = train_test_split(X, Y, test_size=0.3, random_state=0)
X_cross, X_tests, Y_cross, Y_tests = train_test_split(X_, Y_, test_size=0.5, random_state=0)
view_train_cross(X_trains, X_cross, Y_trains, Y_cross)


"""## model training"""
# import then load baseline model architecture
model = load_baseline()

# begin model training
history = model.fit(
    X_trains, Y_trains,
    epochs=100,
    validation_data=(X_cross, Y_cross)
)

# extract the history of accuracy and cost of model
results = {
    'train_loss': history.history['loss'],
    'train_binary_crossentropy': history.history['binary_crossentropy'],
    'train_binary_accuracy': history.history['binary_accuracy'],
    'cross_val_loss': history.history['val_loss'],
    'cross_val_binary_crossentropy': history.history['val_binary_crossentropy'],
    'cross_val_binary_accuracy': history.history['val_binary_accuracy']
}

"""## results visualization"""
figure = plt.figure(figsize=(15, 10))
axis = figure.add_subplot()

styles = [('p:', '#5d42f5'), ('h-', '#fc03a5'), ('o:', '#1e8beb'), ('x--','#1eeb8f'), ('+--', '#0eb802'), ('8-', '#f55600')]

for index, (key, value) in enumerate(results.items()):
  axis.plot(np.array(history.epoch) + 1, value, styles[index][0] ,color=styles[index][1], alpha=0.5, label=key)

axis.set_ylabel('metric value')
axis.set_xlabel('epochs')
axis.legend()
plt.savefig('./figures/breast cancer classifier train and dev results.png')
plt.show()

# save baseline model
model.save('./models/baseline_model.h5')

# save results of trained model
with open("./results/baseline_model_results.json", "w") as out_file:
  json.dump(results, out_file)