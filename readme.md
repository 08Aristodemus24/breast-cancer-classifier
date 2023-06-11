# **STILL IN PRODUCTION**

# Usage
create virtual or conda environment (if conda use conda create -n <env name> python=3.10) enter environment (if conda use conda activate <env name>) in current working directory containing requirements.txt use pip install -r requirements.txt

## to implement:
**1st PHASE: building tuned model**
- train multiple models with different hyperparameters with original dataset
- pick out the best model with the use of different hyperparameters
- once ensemble model training to extract best hyper params extract best params and save to file

**2nd PHASE: selecting the features**
- the aco algorithm will use either the baseline or tuned model to extract the best features
- model w/default hyper params or baseline and model w/ best hyper params or tuned can both be used to select the features
- <u>extract the cost of the training of a model at the last epoch because by then weights are optimized. Do this in classifier-script.py</u>
- <u>extract cross validation cost using model at last epoch as well since by then weights are optimized. Do this in classifier-script.py</u>
- <u>once best ant_colony is ran and along with it all the nn training sub processes, extract all ants in each iteration and see what ant contains the best costs</u>
- <u>pick out the best ant and identify its features or the feature indeces it has taken as its path</u>

- <u>show graphs of best ants in each epoch</u>
- <u>create graph in such a way that the label is named after the ant at iteration i</u>
- important details to return are **the features that the best ant used** and **what the graph looks like for this ant**

- <u>need to figure out how to accumulate the train_loss, train_accuracy, and train binary_cross_entropy at each run</u>
- add early stopping since cross validation cost tends to go up on some occassions as epochs go by
- <u>why isn't train loss and cross validation loss not being shown?</u>
- <u>implement saving all the best ants at each epoch to a file</u>
- problem to solve is the fact that results in reduced dataset is significantly less better than the original dataset, when it must be the case that the reduced dataset ought to have better results


**3rd PHASE: training baseline model and tuned model**
- train baseline model on select features and in original dataset
- train tuned model on select features and in original dataset


**questions to ask**
- what is the performance of the baseline model using the original dataset
- what is the performance of the baseline model using the selected features
- what is the performance of the tuned model using the original dataset
- what is the performance of the tuned model using the selected features

## Citations (ACO algorithm is based on the following papers):
1. Mousavi, Seyed Muhammad Hossein, S. Younes MiriNezhad, and Mir Hossein Dezfoulian. "Galaxy gravity optimization (GGO) an algorithm for optimization inspired by comets life cycle." 2017 Artificial Intelligence and Signal Processing Conference (AISP). IEEE, 2017. Hope it help you (Be Happy :)
- Feature Extraction and Feature Selection are two different tasks. Feature
- Extraction is initial and vital step, but feature selection is optional.
- There are lots of evolutionary feature selection code are online for
- MATLAB but not feature extraction, especially for image. This code extracts
- features out of 10 classes of images with Ant Colony Optimization (ACO)
- evolutionary algorithm and compared it with extracted features using
- SURF with KNN classifier. Dataset is consists of 100 samples of small
- objects in 10 classes. You can use your data but labeling is done manually
- which you have to change it. following parameters are so important which
- you have to play with them in order to get desired results. Parameters
- are: 'nf', 'MaxIt', 'nAnt', knn classifier neighbors and number of hidden
- layers in "TrainNN.m" file. 

Feel free to contact me:
Email: mosavi.a.i.buali@gmail.com
Author: Seyed Muhammad Hossein Mousavi
My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916
My GitHub: https://github.com/SeyedMuhammadHosseinMousavi

https://www.wikihow.tech/Implement-a-Stack-Data-Structure-in-C%2B%2B
