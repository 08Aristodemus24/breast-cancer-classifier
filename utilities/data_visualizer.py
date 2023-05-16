import pandas as pd

def view_data_info(df):
    Y = df['diagnosis']
    X = df.loc[:, df.columns != 'diagnosis']

    print(df.head())
    print(df.shape)
    print(df.columns[0:32])
    print(df.loc[:, df.columns != 'Unnamed: 32'])

    print(X.head())
    print(Y.head())
    print(X.shape)
    print(Y.shape)
    
def view_train_cross(X_trains, X_cross, Y_trains, Y_cross):
    print(X_trains.shape)
    print(Y_trains.shape)
    print(X_cross.shape)
    print(Y_cross.shape)
    print(X_trains)
    print(Y_trains)
    print(X_trains.dtypes)