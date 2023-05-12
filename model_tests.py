import tensorflow as tf
import numpy as np
from baseline_model_train import X_tests, Y_tests

if __name__ == "__main__":
    baseline_model = tf.keras.models.load_model('./models/baseline_model.h5')
    # baseline_model.summary()
    results = baseline_model.evaluate(X_tests, Y_tests)
    print(results)

    logits = baseline_model.predict(X_tests)
    Y_preds = (tf.nn.sigmoid(logits).numpy() >= 0.5).astype(int)

    print(Y_preds)


    