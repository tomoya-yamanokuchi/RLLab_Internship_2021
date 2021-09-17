import os
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
        assert config.vae.latent_dim > 0
        super(VariationalAutoEncoder, self).__init__()

        self.config    = config
        self.alpha     = config.vae.alpha
        self.beta      = config.vae.beta
        self.encoder   = Encoder(config.vae.latent_dim, config.vae.input_shape).construct()
        self.decoder   = Decoder(config.vae.latent_dim).construct()
        self.objective = Loss()

        self.total_loss_tracker                = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker       = keras.metrics.Mean(name="reconstruction_loss")
        self.alpha_reconstruction_loss_tracker = keras.metrics.Mean(name="alpha_reconstruction_loss_tracker")
        self.kl_loss_tracker                   = keras.metrics.Mean(name="kl_loss")
        self.beta_kl_loss_tracker              = keras.metrics.Mean(name="beta_kl_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z      = self.encoder(data)
            reconstruction            = self.decoder(z)
            reconstruction_loss       = self.objective.reconstruction(data, reconstruction)
            alpha_reconstruction_loss = self.alpha * reconstruction_loss
            kl_loss                   = self.objective.kullback_leibler(z_mean, z_log_var)
            beta_kl_loss              = self.beta * kl_loss
            total_loss                = alpha_reconstruction_loss + beta_kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.alpha_reconstruction_loss_tracker.update_state(alpha_reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.beta_kl_loss_tracker.update_state(beta_kl_loss)
        return {
            "loss"                              : self.total_loss_tracker.result(),
            "reconstruction_loss"               : self.reconstruction_loss_tracker.result(),
            "alpha_reconstruction_loss_tracker" : self.alpha_reconstruction_loss_tracker.result(),
            "kl_loss"                           : self.kl_loss_tracker.result(),
            "beta_kl_loss"                      :  self.beta_kl_loss_tracker.result(),
        }
