import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class BaselineNet():
    def __init__(self, input_size, output_size):
        self.model = keras.Sequential(
            layers=[
                keras.Input(shape=(input_size,)),
                layers.Dense(64, activation="relu", name="relu_layer"),
                layers.Dense(output_size, activation="linear", name="linear_layer")
            ],
            name="baseline")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-2)

    def forward(self, observations):
        output = tf.squeeze(self.model(observations))
        return output

    def update(self, observations, target):
        with tf.GradientTape() as tape:
            predictions = self.forward(observations)
            loss = tf.keras.losses.mean_squared_error(y_true=target, y_pred=predictions)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))








