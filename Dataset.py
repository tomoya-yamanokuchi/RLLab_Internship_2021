from abc import ABCMeta, abstractmethod

'''
    データセットの作成する際にはこのDataseクラス（抽象クラス）を継承してください
    （「MNISTDataset」クラスを参考に作ってみてください）
'''

class Dataset(metaclass=ABCMeta):
    @abstractmethod
    def load_train(self):
        pass

    @abstractmethod
    def load_test(self):
        pass
