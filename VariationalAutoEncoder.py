from tensorflow.python.util.tf_export import API_ATTRS
from Loss import Loss
from Decoder import Decoder
from Encoder import Encoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class VariationalAutoEncoder(keras.Model):
    def __init__(self, config):
        assert config.encoder.latent_dim == config.decoder.latent_dim
        super(VariationalAutoEncoder, self).__init__()

        self.config    = config
        self.encoder   = Encoder(config.encoder).construct()
        self.decoder   = Decoder(config.decoder).construct()
        self.objective = Loss()

        self.total_loss_tracker          = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker             = keras.metrics.Mean(name="kl_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction       = self.decoder(z)
            reconstruction_loss  = self.objective.reconstruction(data, reconstruction)
            kl_loss              = self.objective.kullback_leibler(z_mean, z_log_var)
            total_loss           = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
