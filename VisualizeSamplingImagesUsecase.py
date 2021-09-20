from Dataset import Dataset
from DatasetFactory import DatasetFactory
import os
import time
from VariationalAutoEncoder import VariationalAutoEncoder
import numpy as np
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from natsort import natsorted
import glob
import cv2


class VisualizeSamplingImagesUsecase:
    def get_color_and_marker(self, label):
        cm              = plt.get_cmap('jet')
        mnist_label_num = 10
        colors          = []
        markers_set     = ["*", "s", "v", "^", "+", "o", "P", "D", "p", "d"]
        markers         = []
        for i in list(label):
            colors.append(cm(1.*i/mnist_label_num))
            markers.append(markers_set[i])
        return colors, markers, cm


    def run(self, config, model_name, save_path):
        vae = VariationalAutoEncoder(config)
        vae.built = True
        vae.load_weights("{}".format(model_load_path))

        # display a n*n 2D manifold of digits
        n          = config.visualize.image_num
        figsize    = config.visualize.figsize
        digit_size = config.visualize.digit_size
        channel    = config.visualize.channel
        # scale      = config.visualize.scale
        z0_range   = config.visualize.z0_range
        z1_range   = config.visualize.z1_range
        figure     = np.zeros((digit_size * n, digit_size * n, channel))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x     = np.linspace(z0_range[0], z0_range[1], n)
        grid_y     = np.linspace(z1_range[0], z1_range[1], n)[::-1]
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = vae.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size, channel)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range    = digit_size // 2
        end_range      = n * digit_size + start_range
        pixel_range    = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="viridis")
        # plt.show()
        os.makedirs(save_path + "/SamplingImages", exist_ok=True)
        # plt.savefig("figure/SamplingImages/{}.png".format(model_name))
        cv2.imwrite(save_path + "/SamplingImages/{}.png".format(model_name), figure*255)



    def plot_given_model(self, config, vae, sampling_range, model_save_path):
        n          = config.image_num
        figsize    = config.figsize
        digit_size = config.digit_size
        figure     = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x     = np.linspace(sampling_range["z0"][0], sampling_range["z0"][1], n)
        grid_y     = np.linspace(sampling_range["z1"][0], sampling_range["z1"][1], n)[::-1]
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = vae.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range    = digit_size // 2
        end_range      = n * digit_size + start_range
        pixel_range    = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 5)
        sample_range_y = np.round(grid_y, 5)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z0")
        plt.ylabel("z1")
        plt.imshow(figure, cmap="Greys_r")
        plt.savefig(model_save_path + "/sampling_images.png")






if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.config_store import ConfigStore

    execution_dir   = os.getcwd()
    config_test     = OmegaConf.load(execution_dir + "/conf/config.yaml").model_load

    if config_test.model_name == "-1":
        save_path       = execution_dir + "/model/" + config_test.model_dir
        abs_model_dir   = execution_dir + "/model/" + config_test.model_dir + "/model"
        path_sub        = sorted(glob.glob(abs_model_dir + "/*.h5"))
        path_sub        = natsorted(path_sub, key=lambda y: y.lower())
        model_load_path = path_sub[-1]
    else:
        model_load_path = execution_dir + "/model/" + config_test.model_dir + "/" + config_test.model_name + ".h5"

    cfg             = OmegaConf.load(execution_dir + "/model/" + config_test.model_dir + "/config.yaml")
    cfg.visualize   = OmegaConf.load(execution_dir + "/conf/config.yaml").visualize
    model_name      = config_test.model_dir + "_" +  config_test.model_name

    usecase = VisualizeSamplingImagesUsecase()
    usecase.run(cfg, model_name, save_path)