import tensorflow as tf
from baseline_model_train import X_tests, Y_tests

if __name__ == "__main__":
    baseline_model = tf.keras.models.load_model('./models/baseline_model.h5')
    baseline_model.summary()
    # baseline.evaluate()
    