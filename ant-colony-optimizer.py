# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utilities.data_preprocessor import preprocess
from utilities.data_visualizer import view_train_cross

from aco_algorithm.ant_colony import Colony


if __name__ == "__main__":
    df = pd.read_csv('./data.csv')
    
    X, Y = preprocess(df)
    X_trains, X_cross, Y_trains, Y_cross = train_test_split(X, Y, test_size=0.3, random_state=0)
    view_train_cross(X_trains, X_cross, Y_trains, Y_cross)

    colony = Colony(X.T, Y, epochs=1)
    colony.run()


