import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Decoder:
    def __init__(self, latent_dim):
        self.latent_dim  = latent_dim

    def construct(self):
        inputs  = keras.Input(shape=(self.latent_dim,))
        x       = layers.Dense(128, activation="relu")(inputs)
        x       = layers.Dense(7 * 7 * 1, activation="relu")(inputs)
        x       = layers.Reshape((7, 7, 1))(x)
        x       = layers.Conv2D(filters=4,  kernel_size=3, activation="relu", strides=1, padding="same")(x)
        x       = layers.Conv2D(filters=16, kernel_size=3, activation="relu", strides=1, padding="same")(x)
        x       = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x       = layers.Conv2D(filters=16, kernel_size=3, activation="relu", strides=1, padding="same")(x)
        x       = layers.Conv2D(filters=32, kernel_size=3, activation="relu", strides=1, padding="same")(x)
        x       = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        outputs = layers.Conv2D(filters=1, kernel_size=3, activation="sigmoid", strides=1, padding="same")(x)
        decoder = keras.Model(inputs, outputs, name="decoder")
        decoder.summary()
        return decoder


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.config_store import ConfigStore


    cfg = OmegaConf.load('/hdd_mount/RLLab_Internship_2021/conf/decoder/decoder.yaml')
    encoder = Decoder(cfg)
