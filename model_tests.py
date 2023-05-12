import tensorflow as tf
# import .

if __name__ == "__main__":
    baseline_model = tf.keras.models.load_model('./models/baseline_model.h5')
    baseline_model.summary()
    # baseline.evaluate()
    