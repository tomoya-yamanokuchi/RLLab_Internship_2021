from matplotlib.pyplot import axis
from VisualizeSamplingImagesUsecase import VisualizeSamplingImagesUsecase
from VisualizeLatentSpaceUsecase import VisualizeLatentSpaceUsecase
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import cv2


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, x_train, y_train, model_save_path, save_decoded_image_num, config_normalize):
        self.x_train                = x_train
        self.y_train                = y_train
        self.model_save_path        = model_save_path
        self.config_normalize       = config_normalize
        self.visLatentSpace         = VisualizeLatentSpaceUsecase()
        self.visSampling            = VisualizeSamplingImagesUsecase()
        self.save_decoded_image_num = save_decoded_image_num
        self.index_save             = list(np.linspace(0, self.x_train.shape[0], save_decoded_image_num, dtype=int, endpoint=False))
        os.makedirs(model_save_path + "/latent_space", exist_ok=True)
        os.makedirs(model_save_path + "/x_true_and_decode", exist_ok=True)


    def on_epoch_end(self, epoch, logs=None):
        z_mean, _, _   = self.model.encoder.predict(self.x_train)
        self.visLatentSpace.plot_given_data(z_mean, self.y_train, self.model_save_path + "/latent_space" + "/latent_space_epoch{}.png".format(epoch))

        x_decoded = self.model.decoder.predict(z_mean[self.index_save])
        self.save_images(self.x_train[self.index_save], x_decoded, epoch)

    def normalize_0_to_1(self, img):
        return (img - self.config_normalize.min_img) / (self.config_normalize.max_img - self.config_normalize.min_img)

    def save_images(self, x_true, x_decoded, epoch):
        x_true    = np.concatenate(np.split(x_true,    indices_or_sections=len(self.index_save), axis=0), axis=2)[0] * 255
        x_decoded = np.concatenate(np.split(x_decoded, indices_or_sections=len(self.index_save), axis=0), axis=2)[0] * 255
        # x_decoded = self.normalize_0_to_1(np.concatenate((x_true, x_decoded), axis=0)) * 255
        # x_decoded = x_decoded*255
        cv2.imwrite(self.model_save_path + "/x_true_and_decode" + "/x_true_and_decode_epoch{}.png".format(epoch), np.concatenate((x_true, x_decoded), axis=0))


