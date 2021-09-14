from MNISTDataset import MNISTDataset


class DatasetFactory:
    def create(self, dataset_name: str):
        assert type(dataset_name) == str

        dataset_dict = {
            "MNIST" : MNISTDataset(),
        }

        print("\n")
        print("======================================")
        print("          dataset = {}       ".format(dataset_name))
        print("======================================")
        print("\n")

        return dataset_dict[dataset_name]


if __name__ == '__main__':
    factory = DatasetFactory()
    factory.create(dataset_name="MNIST")