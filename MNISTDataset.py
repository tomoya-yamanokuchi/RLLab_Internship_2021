from Dataset import Dataset
from tensorflow import keras
import GeneralService as srv

'''
    データセットをロードするクラスはDataseクラス（抽象クラス）を継承してください
'''

class MNISTDataset(Dataset):
    def load_train(self):
        (x_train, y_train), _ = keras.datasets.mnist.load_data()
        x_train = srv.format_image(x_train)
        return x_train, y_train

    def load_test(self):
        _, (x_test, y_test) = keras.datasets.mnist.load_data()
        x_test = srv.format_image(x_test)
        return x_test, y_test