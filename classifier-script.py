
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class Ant:
    def __init__(self):
        self._tour = []
        self._cost = []
        self._out = []

    def __str__(self):
        return f"""
        tours: {self.tour}\n
        costs: {self.cost}\n
        outs: {self.out}\n
        """
        
    @property
    def tour(self):
        return self._tour
    
    def append_tour(self, val):
        self._tour.append(val)

    @property
    def cost(self):
        return self._cost
    
    def append_cost(self, val):
        self._cost.append(val)

    @property
    def out(self):
        return self._out
    
    def append_out(self, val):
        self._out.append(val)
        

class Colony:
    def __init__(self, X, Y, epochs=15, num_sampled_features=15, num_ants=3, Q=1, tau_0=1, alpha=1, beta=1, rho=0.05):
        # must be a 1024 x 100 matrix
        self.X = X
        
        # must be 1 x 100 matrix
        self.Y = Y
        
        # 1024 features
        self.num_features = X.shape[0]
        
        # 100 instances
        self.num_instances = X.shape[1]
        
        # desired number of selected feaures
        self.num_sampled_features = num_sampled_features
        
        # ACO algorithm hyper parameters
        self.epochs = epochs
        self.num_ants = num_ants
        self.Q = Q
        
        # initial intensity of pheromone values in pheromone matrix 'tau'
        self.tau_0 = tau_0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        
        # initialize heuristic info matrix to be 1024 x 1024
        self.eta = np.ones((X.shape[0], X.shape[0]))
        
        # init pheromone matrix to be 1024 x 1024
        # multiplied by initialized tau_0 value
        self.tau = tau_0 * np.ones((X.shape[0], X.shape[0]))
        
        # list to hold best cost values out of all ants in each iteration
        # e.g. ant 1 out of all ants holds best path/cost of iteration/epoch 1
        self.best_cost = []
        self.ants = np.empty(shape=(num_ants, 1), dtype=np.dtype(Ant))
        
        # initially best ant cost is an infinite value
        self.best_ant_cost = np.inf
        
    def run(self):
        # loop from 0 to 14
        for epoch in range(self.epochs):
            
            # move ants
            print(f'epoch {epoch}')
            
            # loop from 0 to 2
            for k in range(self.num_ants):
                
                # instantiate an Ant object
                temp_ant = Ant()
                
                # since we have 1024 features for ex, generate a random
                # number from 0 to 1023 inclusively, 1024 is excluded
                temp_tour = np.random.randint(0, self.num_features)
                temp_ant.append_tour(temp_tour)
                self.ants[k, 0] = temp_ant
                
                # loop from 1 to 1023, instead of 0 to 1023, but stop at 1024
                for l in range(1, self.num_features):
                    
                    # since we are accessing last element of tour
                    # attribute of ant make sure, .tour is never 
                    # empty or statemetn will raise error
                    i = self.ants[k, 0].tour[-1]
                    
                    # P when calculated is a 1 x 1024 row vector
                    # or will always be a 1 x num_features row vector
                    P = np.power(self.tau[i, :], self.alpha) * np.power(self.eta[i, :], self.beta)
                    
                    
                    # sets the visited spots of the ants in the P matrix to 0
                    # e.g. [1000] accesses P[[1000]], or element at 1000th index
                    # [1000, 241] accesses elements at 1000th and 241st index and
                    # sets them to 0
                    P[self.ants[k, 0].tour] = 0
                    
                    # sum all elements in P row vector and use as denominator
                    P = P / np.sum(P)
                    
                    j = self.roulette(P)
                    self.ants[k, 0].append_tour(j)
                
                print(self.ants[k, 0])
                    
    def roulette(self, P):
        # generate random float between (0, 1) exclusively
        r_num = np.random.uniform()
        
        # since P is a 1 x num_features matrix
        # np.cumsum(P) will be same shape as P
        p_cum_sum = np.cumsum(P)
        
        bools = (r_num <= p_cum_sum).astype(int)
        
        # return the index of the first occurence of 
        # a true/1 value in the bools array 
        return np.where(bools == 1)[0][0]
        
        
                
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

if __name__ == "__main__":

    df = pd.read_csv('./data.csv')
    
    # delete id diagnosis
    Y = df['diagnosis']

    # transform Y to 2-dim 1 x m matrix
    Y = Y.to_numpy().reshape(Y.shape[0], -1)

    X = df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1, inplace=False)
    # view_data_info(df)

    X_trains, X_cross, Y_trains, Y_cross = train_test_split(X, Y, test_size=0.3, random_state=0)
    # view_train_cross(X_trains, X_cross, Y_trains, Y_cross)    

    colony = Colony(X_trains.T, Y_trains, epochs=1)
    colony.run()


