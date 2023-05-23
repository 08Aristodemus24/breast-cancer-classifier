import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def train_cross_results(curr_epoch, curr_ant, results, visualize=True, epochs=100):
    # use matplotlib backend
    mpl.use('Agg')

    print(f'results: {results}\n')
    figure = plt.figure(figsize=(15, 10))
    axis = figure.add_subplot()

    styles = [('p:', '#5d42f5'), ('h-', '#fc03a5'), ('o:', '#1e8beb'), ('x--','#1eeb8f'), ('+--', '#0eb802'), ('8-', '#f55600')]

    for index, (key, value) in enumerate(results.items()):
        axis.plot(np.arange(epochs) + 1, value, styles[index][0] ,color=styles[index][1], alpha=0.5, label=key)

    axis.set_title(f'ant {curr_ant} at epoch {curr_epoch}')
    axis.set_ylabel('metric value')
    axis.set_xlabel('epochs')
    axis.legend()

    # showonly if visualize arg is true
    if visualize is True:
        plt.show()

    # save figure
    plt.savefig(f'./figures/ant {curr_ant} at epoch {curr_epoch}.png')

    # delete figure
    del figure


def train_cross_results_v2(results, epochs=100):
    # use matplotlib backend
    mpl.use('Agg')

    figure = plt.figure(figsize=(15, 10))
    axis = figure.add_subplot()

    styles = [('p:', '#5d42f5'), ('h-', '#fc03a5'), ('o:', '#1e8beb'), ('x--','#1eeb8f'), ('+--', '#0eb802'), ('8-', '#f55600')]

    for index, (key, value) in enumerate(results.items()):
        axis.plot(np.arange(epochs) + 1, value, styles[index][0] ,color=styles[index][1], alpha=0.5, label=key)

    axis.set_ylabel('metric value')
    axis.set_xlabel('epochs')
    axis.legend()

    plt.show()
    plt.savefig('./figures/baseline breast cancer classifier train and dev results.png')

    # delete figure
    del figure
    

    