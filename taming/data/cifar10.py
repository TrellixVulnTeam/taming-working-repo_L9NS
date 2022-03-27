import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


class CIFAR10Train(Dataset):
    def __init__(self, flip_p=0.5):
        super().__init__()
        dset = torchvision.datasets.CIFAR10("data/cifar10", train=True, download=True,
                                            transform=transforms.Compose([transforms.RandomHorizontalFlip(p=flip_p),
                                                                          transforms.ToTensor()])
                                            )
        self.data = dset
        self.size = 32

    def __getitem__(self, i):
        x, y = self.data[i]
        x = 2.0 * x - 1.
        x = x.permute(1, 2, 0)
        return {"image": x,
                "class": y}

    def __len__(self):
        return len(self.data)


class CIFAR10Validation(Dataset):
    def __init__(self):
        super().__init__()
        dset = torchvision.datasets.CIFAR10("data/cifar10", train=False, download=True,
                                            transform=transforms.ToTensor())
        self.data = dset
        self.size = 32

    def __getitem__(self, i):
        x, y = self.data[i]
        x = 2.0*x - 1.
        x = x.permute(1,2,0)
        return {"image": x,
                "class": y}

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    d = CIFAR10Validation()
    ex = d[0]
    print(ex["image"].shape)