import tensorflow as tf
import numpy as np
from baseline_model_train import X_tests as X_tests_base, Y_tests+ 
from sklearn.metrics import confusion_matrix

import seaborn as sb
import matplotlib.pyplot as plt

import sys

def accuracy(model, X, Y):
    # number of instances of input X
    m = X.shape[0]

    # make predictions
    logits = model.predict(X)
    Y_preds = (tf.nn.sigmoid(logits).numpy() >= 0.5).astype(int)

    # calculate how many predictions were right
    acc = np.sum((Y_preds == Y) / m)
    results = 'Accuracy: {:.2%}'.format(acc)

    return acc, results


# this is the script where both produced baseline and tuned modesl by model_trainining notebook can be tested
if __name__ == "__main__":
    if sys.argv[1].lower() == "baseline":
        baseline_model = tf.keras.models.load_model('./trained_models/baseline_model.h5')
        
        # evaluate on test dataset the trained model
        # print loss, binary cross entropy cost, and accuracy
        results = baseline_model.evaluate(X_tests, Y_tests)
        print(results)

        res_acc = accuracy(baseline_model, X_tests, Y_tests)
        print(res_acc[1])

        # make predictions and see confusion matrix
        logits = baseline_model.predict(X_tests)
        Y_preds = (tf.nn.sigmoid(logits).numpy() >= 0.5).astype(int)
        conf_matrix = confusion_matrix(Y_tests, Y_preds)
        sb.heatmap(conf_matrix, annot=True)
        plt.show()

    else:
        tuned_model = tf.keras.models.load_model('./trained_models/tuned_model.h5')

        results = tuned_model.evaluate()


    