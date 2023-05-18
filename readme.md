# **STILL IN PRODUCTION**

# Usage
create virtual or conda environment (if conda use conda create -n <env name> python=3.10) enter environment (if conda use conda activate <env name>) in current working directory containing requirements.txt use pip install -r requirements.txt

## to implement:
- <u> build baseline model </u>
- build better tuned model
- explore data and see graphs
- <u>extract the cost of the training of a model at the last epoch because by then weights are optimized. Do this in classifier-script.py</u>
- <u>extract cross validation cost using model at last epoch as well since by then weights are optimized. Do this in classifier-script.py</u>

- once best ant_colony is ran and along with it all the nn training sub processes, extract all ants in each iteration and see what ant contains the best costs
- pick out the best ant and identify its features
- train the baseline model with these features again
- train multiple models with different hyperparameters with these features
- pick out the best model with the use of different hyperparameters

- show graphs of best ants in each epoch
- create graph in such a way that the label is named after the ant at iteration i
- important details to return are **the features that the best ant used** and **what the graph looks like for this ant**

- <u>need to figure out how to accumulate the train_loss, train_accuracy, and train binary_cross_entropy at each run</u>
- add early stopping since cross validation cost tends to go up on some occassions as epochs go by
- why isn't train loss and cross validation loss not being shown