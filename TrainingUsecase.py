from Dataset import Dataset
from DatasetFactory import DatasetFactory
import os
import time
from VariationalAutoEncoder import VariationalAutoEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint


class TrainingUsecase:
    def run(self, config, model_save_path):
        vae = VariationalAutoEncoder(config)

        vae.compile(optimizer=keras.optimizers.Adam(
            learning_rate = config.optimzer.lr,
            beta_1        = config.optimzer.beta_1,
            beta_2        = config.optimzer.beta_2,
            epsilon       = None if config.optimzer.epsilon=='None' else config.optimzer.epsilon,
            decay         = config.optimzer.decay,
            amsgrad       = config.optimzer.amsgrad
        ))

        factory    = DatasetFactory()
        dataset    = factory.create(config.dataset.dataset_name)
        x_train, _ = dataset.load_train()


        checkpoint = ModelCheckpoint(
            filepath          = os.path.join(model_save_path, 'model_epoch{epoch:03d}.h5'),
            monitor           = config.checkpoint.monitor,
            save_weights_only = config.checkpoint.save_weights_only,
            save_best_only    = config.checkpoint.save_best_only,
            save_freq         = config.checkpoint.save_freq,
            verbose           = config.checkpoint.verbose,
        )

        vae.fit(
            x          = x_train,
            epochs     = config.optimzer.epochs,
            batch_size = config.optimzer.batch_size,
            callbacks  = [checkpoint],
        )



if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.config_store import ConfigStore
    import GeneralService as srv

    execution_dir = os.getcwd()

    @hydra.main(config_path="conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        model_save_path = srv.create_model_save_path(cfg, execution_dir)
        os.makedirs(model_save_path, exist_ok=True)
        OmegaConf.save(config=cfg, f=model_save_path+"/config.yaml")
        usecase = TrainingUsecase()
        usecase.run(cfg, model_save_path)

    get_config()