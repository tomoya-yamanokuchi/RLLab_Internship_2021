from Dataset import Dataset
from tensorflow import keras
import GeneralService as srv

# Additional import packages
import os, glob
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import cv2

'''
    データセットをロードするクラスはDataseクラス（抽象クラス）を継承してください
'''

class soilDataset(Dataset):
    def load_train(self):
        # label: 0-4
        soil_type = ["clay", "gravel", "loam", "sand"]
        X = [] # data
        Y = [] # label
        for index, dir_name in enumerate(soil_type):
            files = glob.glob("../../../data/" + dir_name + '/*.png')
            for f in files:
                # open image
                image = cv2.imread(f)
                # convert image to numpy array
                data = np.asarray(image)
                # append
                X.append(data) # data: image
                Y.append(index) # label: soil type
        X = np.array(X)
        Y = np.array(Y)
        # split dataset
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        # change dim (add "1" channel)
        x_train = srv.format_image(x_train)
        return x_train, y_train


    def load_test(self):
        # label: 0-4
        soil_type = ["clay", "gravel", "loam", "sand"]
        X = [] # data
        Y = [] # label
        for index, dir_name in enumerate(soil_type):
            files = glob.glob("data/" + dir_name + '/*.png')
            for f in files:
                # open image
                image = Image.open(f)
                # convert image to numpy array
                data = np.asarray(image)
                # append
                X.append(data) # data: image
                Y.append(index) # label: soil type
        X = np.array(X)
        Y = np.array(Y)
        print(X)
        # split dataset
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        # change dim (add "1" channel)
        x_test = srv.format_image(x_test)
        return x_test, y_test