import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from settings import logger


class GenerateData:
    
    def __init__(self, data_dir, split_ratio):
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.params = {'batch_size': 16,
                       'shuffle': True,
                       'num_workers': 4}
    
    def image_dataset(self):
        image_data = torchvision.datasets.ImageFolder(
            root=self.data_dir,
            transform=transforms.Compose([
                transforms.Resize((32,32)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
        )
        logger.info(f'Dataset sample length: {len(image_data)}')
        return image_data

    def split_dataset(self):
        image_dataset = self.image_dataset()
        num_dataset = len(image_dataset)
        indices = list(range(num_dataset))
        split_idx = int(np.floor(self.split_ratio*num_dataset))
        np.random.shuffle(indices)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        train_dataset = Subset(image_dataset, indices=train_idx)
        test_dataset = Subset(image_dataset, indices=test_idx)
        logger.info(f'Train data size: {len(train_dataset)}')
        logger.info(f'Test data size: {len(test_dataset)}')
        return (train_dataset, test_dataset)
        
    def load_generators(self):
        train_dataset, test_dataset = self.split_dataset()
        train_gen = DataLoader(train_dataset, **self.params)
        test_gen = DataLoader(test_dataset, **self.params)
        logger.info(f'Train generator size: {len(train_gen)}')
        logger.info(f'Test generator size: {len(test_gen)}')
        return (train_gen, test_gen)
    