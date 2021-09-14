from Dataset import Dataset
from DatasetFactory import DatasetFactory
import os
import time
from VariationalAutoEncoder import VariationalAutoEncoder
import numpy as np
from tensorflow import keras
from keras.callbacks import ModelCheckpoint


class VisualizeLatentSpaceUsecase:
    def run(self, config, model_load_path):
        # vae = keras.models.load_model("{}.h5".format(model_load_path))

        vae = VariationalAutoEncoder(config)
        # vae.load_weights("{}.h5".format(model_load_path))
        vae.compile(optimizer=keras.optimizers.Adam(
            learning_rate = config.optimzer.lr,
            beta_1        = config.optimzer.beta_1,
            beta_2        = config.optimzer.beta_2,
            epsilon       = None if config.optimzer.epsilon=='None' else config.optimzer.epsilon,
            decay         = config.optimzer.decay,
            amsgrad       = config.optimzer.amsgrad
        ))
        vae.load_weights("{}.h5".format(model_load_path))

        # factory = DatasetFactory()
        # dataset = factory.create(config.dataset.dataset_name)
        # x_train = dataset.load_train()



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