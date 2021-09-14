import numpy as np


def format_image(img): 
    img = img.astype("float32")
    if img.max() > 1.0: 
        img = img / 255.0

    if len(img.shape) == 3: 
        img = np.expand_dims(img, -1)

    return img