import numpy as np
import time
import os

def format_image(img):
    img = img.astype("float32")
    if img.max() > 1.0:
        img = img / 255.0

    if len(img.shape) == 3:
        img = np.expand_dims(img, -1)

    return img



def create_model_save_path(config, execution_dir):
    use_param = [
        "epoch{}".format(config.optimzer.epochs),
        "batch_size{}".format(config.optimzer.batch_size),
        "dataset_{}".format(config.dataset.dataset_name),
        time.strftime('%Y%m%d%H%M%S', time.localtime())
    ]
    model_name = "_".join(use_param)
    model_save_path = execution_dir + "/model/" + model_name
    return model_save_path
