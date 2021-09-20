import os
import shutil
import glob
from traceback import print_list
from PIL import Image
import cv2
import copy
import numpy as np
from natsort import natsorted

class ResizeSoilImage_with_colormap:
    def __init__(self):
        self.dst_dir        = "data/"
        self.type_soil_list = ["clay", "gravel", "loam", "sand"]
        self.size_width     = 64
        self.size_height    = 64

    def remove_directory(self):
        [shutil.rmtree(self.dst_dir + type_soil) for type_soil in self.type_soil_list]

    def make_directory(self):
        [os.makedirs(self.dst_dir + type_soil) for type_soil in self.type_soil_list]

    def reshape(self, img):
        width = img.shape[1]
        w     = 30
        img   = img[:, 0+w:width-w] # height, width
        img   = cv2.resize(img, (self.size_width, self.size_height))
        return img

    def _check_minimum(self, img):
        if self.min_img > img.min():
            self.min_img = copy.deepcopy(img.min())
            print(" min : {}".format(self.min_img))

    def _check_maximum(self, img):
        if self.max_img < img.max():
            self.max_img = copy.deepcopy(img.max())
            print(" max : {}".format(self.max_img))

    def get_minmax_value_of_image(self):
        self.min_img = 999
        self.max_img = 0
        for type_soil in self.type_soil_list:
            files = glob.glob('./soil_data/' + type_soil +  '/*2.png') # only 3rd excavation
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                img = self.reshape(img)
                self._check_minimum(img)
                self._check_maximum(img)


    def normalize_0_to_1(self, img):
        return (img - self.min_img) / (self.max_img - self.min_img)

    def _create_save_path(self, f, type_soil):
        root, ext = os.path.splitext(f)
        basename  = os.path.basename(root)
        path      = self.dst_dir + type_soil + "/" + basename + ext
        return path


    def load_and_preprocess_and_save(self):
        N_save_per_type = 100
        save_images     = []
        for type_soil in self.type_soil_list:
            print(" type_soil : {}".format(type_soil))
            files        = glob.glob('./soil_data/' + type_soil +  '/*2.png') # only 3rd excavation
            files        = natsorted(files)
            save_index   = list(np.linspace(0, len(files)-1, N_save_per_type, dtype=int))
            _save_images = []
            for i, f in enumerate(files):
                img  = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                img  = self.reshape(img)
                img  = self.normalize_0_to_1(img) * 255
                img  = self.applyColorMap(img)
                path = self._create_save_path(f, type_soil)
                cv2.imwrite(path, img)
                if i in save_index:
                    _save_images.append(img)
            _save_images = np.concatenate(_save_images, axis=1)
            save_images.append(_save_images)
        save_images = np.concatenate(save_images, axis=0)
        print(" save_images = {}".format(save_images.shape))
        cv2.imwrite(self.dst_dir + "/x_train_only_3step.png", save_images)


    def applyColorMap(self, img):
        '''
        - The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.
        - The result is the colormapped source image.
        '''
        img = img.astype(np.uint8)
        return cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)


if __name__ == '__main__':
    resize = ResizeSoilImage_with_colormap()
    resize.remove_directory()
    resize.make_directory()
    resize.get_minmax_value_of_image()
    resize.load_and_preprocess_and_save()



