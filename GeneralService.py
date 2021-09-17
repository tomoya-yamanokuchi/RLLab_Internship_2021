import numpy as np
import time
import os

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



def format_image(img):
    img = img.astype("float32")
    if img.max() > 1.0:
        img = img / 255.0

    if len(img.shape) == 3:
        img = np.expand_dims(img, -1)

    return img



def create_model_save_path(config, execution_dir):
    use_param = [
        "epoch{}".format(config.optimizer.epochs),
        "batch_size{}".format(config.optimizer.batch_size),
        "dataset_{}".format(config.dataset.dataset_name),
        time.strftime('%Y%m%d%H%M%S', time.localtime())
    ]
    model_name = "_".join(use_param)
    model_save_path = execution_dir + "/model/" + model_name
    return model_save_path


def plot_loss_history(loss_history, model_save_path):
    fig, ax = plt.subplots(len(loss_history), 1)
    for i, (key, val) in enumerate(loss_history.items()):
        print(" key : {}".format(key))
        ax[i].plot(np.array(val) , label=key)
    plt.legend()
    plt.savefig(model_save_path + "/loss.png")