import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers


class PolicyNet():
    def __init__(self, input_size, output_size):
        self.model = keras.Sequential(
            layers=[
                keras.Input(shape=(input_size,)),
                layers.Dense(64, activation="relu", name="relu_layer"),
                layers.Dense(output_size, activation="linear", name="linear_layer")
            ],
            name="policy")

    def action_distribution(self, observations):
        logits = self.model(observations)
        return tfp.distributions.Categorical(logits=logits)

    def sampel_action(self, observations):
        sampled_actions = self.action_distribution(observations).sample().numpy()
        return sampled_actions






