import numpy as np # linear algebra
from sklearn.model_selection import train_test_split

from models.baseline_model_arc import load_baseline
from utilities.data_visualizer import view_train_cross, train_cross_results
from tensorflow.keras.callbacks import EarlyStopping

class Ant:
    def __init__(self):
        self._tour = []
        self._cost = np.inf
        self._output = np.inf

    def __str__(self):
        return f"""
        tours: {self.tour}\n
        length: {len(self.tour)}\n

        costs: {self.cost}\n
        length: {len(self.tour)}\n

        outputs: {self.output}\n
        length: {len(self.tour)}\n
        """
        
    @property
    def tour(self):
        return self._tour
    
    def append_tour(self, val):
        self._tour.append(val)

    @property
    def cost(self):
        return self._cost
    
    @cost.setter
    def cost(self, val):
        self._cost = val

    @property
    def output(self):
        return self._output
    
    @output.setter
    def output(self, val):
        self._output = val
        

class Colony:
    def __init__(self, X, Y, epochs=15, num_sampled_features=15, num_ants=3, Q=1, tau_0=1, alpha=1, beta=1, rho=0.05):
        # X must be a 1024 x 100 matrix and Y must be 1 x 100 matrix
        self.X = X
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
        self.best_ants = []

        self.ants = np.empty(shape=(num_ants, 1), dtype=np.dtype(Ant))
        
        # initially best ants cost is an infinite value
        self.best_ant = Ant()
        
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
                
                # loop from [1] to [1023], instead of [0] to [1023], but stop at 1024
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
                

                
                # calculate cost given the paths made by the and
                cost, output = self.J(epoch, k, self.ants[k, 0].tour, self.num_sampled_features, {
                    'X': self.X,
                    'Y': self.Y,
                    'num_features': self.num_features,
                    'num_instances': self.num_instances
                }) 

                self.ants[k, 0].cost = cost
                self.ants[k, 0].output = output

                # use current cost of ant k at iteration i and compare
                # to current best ant cost
                if self.ants[k, 0].cost < self.best_ant.cost:
                    self.best_ant = self.ants[k, 0]

            # updating pheromones for positive feedback
            for k in range(self.num_ants):
                # append the first node to the whole path made by ant
                tour = np.append(self.ants[k, 0].tour, self.ants[k, 0].tour[0])

                # go through now all features from index [0] to [1023] 
                for l in range(self.num_features):
                    i = tour[l]
                    j = tour[l + 1]
                    self.tau[i, j] = self.tau[i, j] + self.Q / self.ants[k, 0].cost

            # updating evaporation rate for negative feedback
            self.tau = (1 - self.rho) * self.tau

            # store the ant with the best cost
            self.best_ants.append(self.best_ant)

            if epoch % 10 == 0:
                print(f'{epoch}\n')

            return self.best_ants

    def J(self, curr_epoch, curr_ant, paths, num_sampled_features, data):
        """paths - is the built path by ant k which is of length 1 to num features - 1 inclusively
        with values 0 to 1023 since indeces of P are used

        num_sampled_features - is the number of features to be 
        sampled which is currently by default 15

        data - is a dictionary of X, Y, num_features, and num_instances"""

        # read data
        X = data['X']
        Y = data['Y']

        # select the paths in q of length 1 to num_features - 1 
        # made by ant k from 1 to nf which recall is 15 by default
        selected_paths = paths[0:num_sampled_features]
        print(f'selected paths: {selected_paths}\n')
        print(f'features selected: {X.columns[selected_paths]}\n')

        # calculate ratio of number of features to 
        # sample to length of tour/path made by ant k
        paths_len = len(paths)
        ratio = num_sampled_features / paths_len

        # recall 1024 x 100 matrix
        # print(X)
        # print(X.index)
        # print(X.iloc[selected_paths].index)
        selected_X = X.iloc[selected_paths]
        # print(f'selected X: {selected_X}\n')

        # train and test error ratios
        train_ratio = 0.7
        cross_ratio = 1 - train_ratio

        # trains a neural network over n_runs times and calculates the average
        # cost in these trained models which mind you is trained on the same dataset
        n_runs = 3
        OVERALL_COST = np.zeros((n_runs,))

        TRAIN_COST = 0.0
        CROSS_VAL_COST = 0.0
        TRAIN_ACC = 0.0
        CROSS_VAL_ACC = 0.0
        TRAIN_LOSS = 0.0
        CROSS_VAL_LOSS = 0.0

        for r in range(n_runs):
            # train the neural network
            results = self.train(selected_X, Y)

            TRAIN_COST += results['train_binary_crossentropy']
            CROSS_VAL_COST += results['cross_val_binary_crossentropy']
            TRAIN_ACC += results['train_binary_accuracy']
            CROSS_VAL_ACC += results['cross_val_binary_accuracy']
            TRAIN_LOSS += results['train_loss']
            CROSS_VAL_LOSS += results['cross_val_loss']

            # calculate overall error in both training 
            # and cross validation datasets
            OVERALL_COST[r] = (train_ratio * results['train_binary_crossentropy'][-1]) + (cross_ratio * results['cross_val_binary_crossentropy'][-1])


        print(f'train cost shape: {TRAIN_COST.shape}')

        # visualize resulting model and get the average of all losses
        # costs, accuracies of all these models values across the number of runs
        train_cross_results(curr_epoch, curr_ant, {
            'train_loss': TRAIN_LOSS / n_runs,
            'train_binary_crossentropy': TRAIN_COST / n_runs,
            'train_binary_accuracy': TRAIN_ACC / n_runs,
            'cross_val_loss': CROSS_VAL_LOSS / n_runs,
            'cross_val_binary_crossentropy': CROSS_VAL_COST / n_runs,
            'cross_val_binary_accuracy': CROSS_VAL_ACC / n_runs
        })

        # calculate the average of all errors or all errors
        # divded by number of training and testing examples
        # calculate and set final cost
        cost = OVERALL_COST.mean()
        print(f'cost: {cost}\n')

        output = {
            'selected_paths': selected_paths,
            'num_sampled_features': num_sampled_features,
            'ratio': ratio,
            'cost': OVERALL_COST,
        }

        return [cost, output]
                    
    def roulette(self, P):
        """P - is the transition probability vector with dimensionality 1 x num_features
        or in this case 1 x 1024 if number of features is 1024
        """
        # generate random float between (0, 1) exclusively
        r_num = np.random.uniform()
        
        # since P is a 1 x num_features matrix
        # np.cumsum(P) will be same shape as P
        p_cum_sum = np.cumsum(P)
        
        bools = (r_num <= p_cum_sum).astype(int)
        
        # return the index of the first occurence of 
        # a true/1 value in the bools array 
        return np.where(bools == 1)[0][0]

    def train(self, X, Y):
        # print(f'selected X shape: {X.shape}\n')
        # print(f'selected Y shape: {Y.shape}\n')
        X_trains, X_, Y_trains, Y_ = train_test_split(X.T, Y.T, test_size=0.3, random_state=0)
        X_cross, X_tests, Y_cross, Y_tests = train_test_split(X_, Y_, test_size=0.5, random_state=0)
        view_train_cross(X_trains, X_cross, Y_trains, Y_cross)

        # import and load baseline model
        model = load_baseline()

        # if cross validation loss does not improve after 10 
        # consecutive epochs we stop training our model early
        # stop_early = EarlyStopping(monitor='val_loss', patience=10)

        # train baseline model
        history = model.fit(
            X_trains, Y_trains,
            epochs=100,
            validation_data=(X_cross, Y_cross),
            # callbacks=[stop_early]
        )

        # extract the history of accuracy and cost of model
        results = {
            'train_loss': np.array(history.history['loss']),
            'train_binary_crossentropy': np.array(history.history['binary_crossentropy']),
            'train_binary_accuracy': np.array(history.history['binary_accuracy']),
            'cross_val_loss': np.array(history.history['val_loss']),
            'cross_val_binary_crossentropy': np.array(history.history['val_binary_crossentropy']),
            'cross_val_binary_accuracy': np.array(history.history['val_binary_accuracy'])
        }

        # return results of model
        return results