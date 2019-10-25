"""everything about data"""

import tarfile
import pickle
from pathlib import Path
from collections import Counter

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms

def cifar_dataloader(
    batch_size=32,
    validation=True,
    train_size=0.8,
    transform=None,
    root='./data/'):
    """
    create and return data loaders for cifar-100

    argument
        batch_size : default 32
            Batch size. This batch size will be used to every
            data loaders that are generated in this function
        validation : default True
            If True, the training dataset will be split to traning and validation.
        train_size : default 0.8
            The size of training dataset.
            Strictly between [0, 1].
            The validation size will be "samples - samples*train_size".
        transform : default None
            The transforms to perform to datasets when loaded.
            Don't forget to Compose before passing the transforms to this function.
        root : default './data/
            The root path of the CIFAR-100 dataset tar.gz file

    return
        train_dataloader
            The data loader for the training dataset
        val_dataloader
            The data loader for the validation dataset
            Only returned when validation arg is True
        test_dataloader
            The data loader for the testing dataset
    """

    test_dataset = CIFAR100(root, train=False, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    trainval_dataset = CIFAR100(root, train=True, transform=transform)

    if validation:
        trainval_samples = len(trainval_dataset)
        train_size       = int(trainval_samples * train_size)
        val_size         = trainval_samples - train_size

        train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True)

        print('train data samples      :', len(train_dataset))
        print('validation data samples :', len(val_dataset))
        print('test data samples       :', len(test_dataset))

        return train_dataloader, val_dataloader, test_dataloader

    train_dataloader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True)

    print('train data samples :', len(trainval_dataset))
    print('test data samples  :', len(test_dataset))

    return train_dataloader, test_dataloader

class CIFAR100(Dataset):
    """
    My original dataset class for CIFAR-100
    """
    def __init__(self, root, train=True, transform=None):
        self._unzip(root)

        if not transform == None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.data = None
        self.targets = None
        self._load_data(root, train)

        self.data_num = self.data.shape[0]

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        if not self.transform == None:
            img = self.transform(img)
        
        return img, target

    def _unzip(self, root):
        """
        If the tar.gz file is not unzipped -> unzip
        """
        root_path = Path(root).resolve()
        cifar_file = root_path / 'cifar-100.tar.gz'
        cifar_folder = root_path / 'cifar-100-python'

        if not cifar_file.exists():
            raise FileNotFoundError(str(cifar_file)+' not found')
        if not cifar_folder.exists():
            try:
                with tarfile.open(str(cifar_file), 'r:gz') as fin:
                    fin.extractall(path=root_path)
            except Exception as e:
                print(e)
                raise

    def _load_data(self, root, train):
        """
        Load data from pickle files
        """
        root_path = Path(root).resolve()
        cifar_folder = root_path / 'cifar-100-python'

        if not cifar_folder.exists():
            raise FileNotFoundError()
        
        meta  = cifar_folder / 'meta'
        if train:
            dataset_file = cifar_folder / 'train'
        else:
            dataset_file = cifar_folder / 'test'

        try:
            with open(str(dataset_file), 'rb') as fin:
                dataset = pickle.load(fin, encoding='bytes')
            self.targets = dataset[b'fine_labels']
            self.data = dataset[b'data'].reshape(-1, 3, 32, 32)
        except Exception as e:
            print(e)
            raise


if __name__=='__main__':
    cifar_dataloader()
    
