from VisualizeSamplingImagesUsecase import VisualizeSamplingImagesUsecase
from VisualizeLatentSpaceUsecase import VisualizeLatentSpaceUsecase
from Dataset import Dataset
from DatasetFactory import DatasetFactory
import os
import time
from VariationalAutoEncoder import VariationalAutoEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import GeneralService as srv



class TrainingUsecase:
    def run(self, config, model_save_path):
        vae = VariationalAutoEncoder(config)

        vae.compile(optimizer=keras.optimizers.Adam(
            learning_rate = config.optimizer.lr,
            beta_1        = config.optimizer.beta_1,
            beta_2        = config.optimizer.beta_2,
            epsilon       = None if config.optimizer.epsilon=='None' else config.optimizer.epsilon,
            decay         = config.optimizer.decay,
            amsgrad       = config.optimizer.amsgrad
        ))

        factory          = DatasetFactory()
        dataset          = factory.create(config.dataset.dataset_name)
        x_train, y_train = dataset.load_train()


        checkpoint = ModelCheckpoint(
            filepath          = os.path.join(model_save_path, 'model_epoch{epoch:03d}.h5'),
            monitor           = config.checkpoint.monitor,
            save_weights_only = config.checkpoint.save_weights_only,
            save_best_only    = config.checkpoint.save_best_only,
            save_freq         = config.checkpoint.save_freq,
            verbose           = config.checkpoint.verbose,
        )

        history_callback = vae.fit(
            x          = x_train,
            epochs     = config.optimizer.epochs,
            batch_size = config.optimizer.batch_size,
            callbacks  = [checkpoint],
        )
        loss_history = history_callback.history
        srv.plot_loss_history(loss_history, model_save_path)


        z_mean, _, _   = vae.encoder.predict(x_train)
        visLatentSpace = VisualizeLatentSpaceUsecase()
        visLatentSpace.plot_given_data(z_mean, y_train, model_save_path)


        visSampling = VisualizeSamplingImagesUsecase()
        sampling_range = {
            "z0" : [z_mean[:, 0].min(), z_mean[:, 0].max()],
            "z1" : [z_mean[:, 1].min(), z_mean[:, 1].max()],
        }
        visSampling.plot_given_model(config.visualize, vae, sampling_range, model_save_path)


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