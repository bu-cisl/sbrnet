import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import yaml


class DataLoader:
    def __init__(self, config):
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.batch_size = config["batch_size"]

    def load_data(self, data_folder):
        train_dataset = torchvision.datasets.MNIST(
            root=data_folder, train=True, transform=self.transform, download=True
        )
        train_loader = data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
        return train_loader
