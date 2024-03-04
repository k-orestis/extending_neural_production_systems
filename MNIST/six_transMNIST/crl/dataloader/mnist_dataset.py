import torch
from torchvision import datasets, transforms

from .base_dataloader import DataLoader

class MNIST(DataLoader):
    def __init__(self, args):
        super(MNIST, self).__init__()
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='../data/mnist', train=True, download=True,     # MNIST dataset already exist in Pytorch 
                           transform=transforms.Compose([                       # So we load it  and we use the root parametre to save the dataset 
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.bsize, shuffle=True)
        self.curr = False

    def initialize_data(self, splits):
        pass

    def reset(self, mode='train', z=None):
        for i, (data, target) in enumerate(self.train_loader):
            return data.view(-1), target

    def get_trace(self):
        return ''

    def change_mt(self):
        pass

def load_mnist_datasets(root, normalize=True, extrap=False):

    if normalize:
        train_dataset = datasets.MNIST(root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        valtest_dataset = datasets.MNIST(root, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    else:
        train_dataset = datasets.MNIST(root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
        valtest_dataset = datasets.MNIST(root, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))

    train_data, train_labels = zip(*train_dataset)
    train_data = torch.stack(train_data)                            # (#images per batch), 1 channel, 28 pixel, 28 pixel
    train_labels = torch.LongTensor(train_labels).unsqueeze(1)      # (#labels per patch), 1

    # now you should divide into groups for validations and test set

    numtest = int(len(valtest_dataset) / 2)
    valtest_data, valtest_labels = zip(*valtest_dataset)
    valtest_data = torch.stack(valtest_data)
    valtest_labels = torch.LongTensor(valtest_labels).unsqueeze(1)

    val_data = valtest_data[:numtest]
    val_labels = valtest_labels[:numtest]

    test_data = valtest_data[numtest:]
    test_labels = valtest_labels[numtest:]

    # Create a dictionary
    mnist_datasets = {
        'train': (train_data, train_labels),    # mnist_datasets["train"]->tuple
        'val': (val_data, val_labels),
        'test': (test_data, test_labels),
    }
    if extrap:
        mnist_datasets['extrapval'] = (test_data, test_labels)
    return mnist_datasets