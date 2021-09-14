import os
import time
from VariationalAutoEncoder import VariationalAutoEncoder
import numpy as np
from tensorflow import keras
import GeneralService as srv
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

        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
        x_train = srv.format_image(x_train)

        checkpoint = ModelCheckpoint(
            filepath          = os.path.join(model_save_path, 'model_epoch{epoch:03d}_loss{loss:.2f}.h5'),
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

    execution_dir   = os.getcwd()
    time_str        = time.strftime('%Y%m%d%H%M%S', time.localtime())
    model_save_path = execution_dir + "/model/" + time_str
    os.makedirs(model_save_path, exist_ok=True)

    @hydra.main(config_path="conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        OmegaConf.save(config=cfg, f=model_save_path+"/config.yaml")
        usecase = TrainingUsecase()
        usecase.run(cfg, model_save_path)

    get_config()