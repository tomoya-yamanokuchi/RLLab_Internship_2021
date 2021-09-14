from Sampler import Sampling
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Encoder: 
    def __init__(self, config):
        self.config      = config
        self.latent_dim  = config.latent_dim
        self.input_shape = config.input_shape
    
    
    def construct(self):
        inputs    = keras.Input(shape=tuple(self.input_shape))
        x         = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
        x         = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x         = layers.Flatten()(x)
        x         = layers.Dense(16, activation="relu")(x)
        z_mean    = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z         = Sampling()([z_mean, z_log_var])
        encoder   = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        return encoder
        

if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.config_store import ConfigStore


    cfg = OmegaConf.load('/hdd_mount/RLLab_Internship_2021/conf/encoder.yaml')
    cfg = cfg['encoder']

    # (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    # mnist_digits = np.concatenate([x_train, x_test], axis=0)
    # mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    
    encoder = Encoder(cfg)
    