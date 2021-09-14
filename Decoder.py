import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Decoder: 
    def __init__(self, config):
        self.config      = config
        self.latent_dim  = config.latent_dim
        # self.input_shape = config.input_shape
    
    
    def construct(self):
        inputs  = keras.Input(shape=(self.latent_dim,))
        x       = layers.Dense(7 * 7 * 64, activation="relu")(inputs)
        x       = layers.Reshape((7, 7, 64))(x)
        x       = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x       = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(inputs, outputs, name="decoder")
        decoder.summary()
        return decoder
    

if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.config_store import ConfigStore


    cfg = OmegaConf.load('/hdd_mount/RLLab_Internship_2021/conf/decoder/decoder.yaml')
    encoder = Decoder(cfg)
    