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


    def run(self, config, model_load_path):
        # vae = keras.models.load_model("{}.h5".format(model_load_path))

        vae = VariationalAutoEncoder(config)
        vae.built = True
        vae.load_weights("{}.h5".format(model_load_path))

        factory = DatasetFactory()
        dataset = factory.create(config.dataset.dataset_name)
        x, y = dataset.load_test()

        z_mean, _, _   = vae.encoder.predict(x)

        colors, makers, cm = self.get_color_and_marker(y)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=colors)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.config_store import ConfigStore

    execution_dir   = os.getcwd()
    config_test     = OmegaConf.load(execution_dir + "/conf/model_load/model_load.yaml")
    model_load_path = execution_dir + "/model/" + config_test.model_dir + "/" + config_test.model_name

    cfg     = OmegaConf.load(execution_dir + "/model/" + config_test.model_dir + "/config.yaml")
    usecase = VisualizeLatentSpaceUsecase()
    usecase.run(cfg, model_load_path)