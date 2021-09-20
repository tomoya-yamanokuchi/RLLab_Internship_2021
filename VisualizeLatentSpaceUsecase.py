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

class VisualizeLatentSpaceUsecase:
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

        factory = DatasetFactory()
        dataset = factory.create(config.dataset.dataset_name)
        x, y = dataset.load_train()

        z_mean, _, _   = vae.encoder.predict(x)

        # colors, makers, cm = self.get_color_and_marker(y)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        # plt.show()
        os.makedirs(save_path + "/latent_space", exist_ok=True)
        plt.savefig(save_path + "/latent_space/{}_{}.png".format(model_name, time.strftime('%Y%m%d%H%M%S', time.localtime())))


    def plot_given_data(self, z_mean, y, model_save_path):
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y)
        plt.colorbar()
        plt.xlabel("z0")
        plt.ylabel("z1")
        plt.savefig(model_save_path)


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

    cfg        = OmegaConf.load(execution_dir + "/model/" + config_test.model_dir + "/config.yaml")
    usecase    = VisualizeLatentSpaceUsecase()
    model_name = config_test.model_dir + "_" +  config_test.model_name
    usecase.run(cfg, model_name, save_path)