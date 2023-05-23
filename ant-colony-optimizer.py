
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utilities.data_preprocessor import preprocess
from utilities.data_visualizer import view_train_cross

from aco_algorithm.ant_colony import Colony

# ## load and preprocess data
df = pd.read_csv('./data.csv')
X, Y = preprocess(df)

colony = Colony(X.T, Y.T, epochs=80, num_ants=20, visualize=False)
best_ants, best_ant = colony.run()

# save each each best ant at each oteration to pkl file

print(*best_ants, sep='\n\n')

# save the overall best ant to pkl file
print('best ant: \n')
print(best_ant)


