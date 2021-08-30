import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from settings import logger


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU())
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Sequential(
            nn.Linear(64*8*8, 128),
            nn.ReLU())
        self.fc2 = nn.Linear(128,2)
        
    def forward(self, inp):
        out = self.layer1(inp)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # Flatten dimension 64*8*8 (2D) to 1*4096 (1D)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out


class TrainNeuralNet:
    
    def __init__(self, dataset=None, net=None, device=None):
        self.dataset = dataset
        self.device = device
        self.net = net.to(self.device)
        self.losses = []
        self.epoch = 1024
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(net.parameters(), lr=0.0001,
                                         momentum=0.9)
        
    def _train_epoch(self, epoch_num):
        for batch, [image, label] in enumerate(self.dataset, 0):
            # Initialize image and label to device (cpu/cuda)
            image = image.to(self.device)
            label = label.to(self.device)
            # Clear old gradients from the last step 
            self.optimizer.zero_grad()
            # Forward the image to NN and read prediction
            predictions = self.net.forward(image)
            # Validate loss
            loss = self.criterion(predictions, label)
            # Back propagate
            loss.backward()
            # Take a step based on the gradients of the parameters
            self.optimizer.step()
            
            if batch%10 == 0:
                logger.info(f'Epoch: {epoch_num+1} - Batch: {batch} - Loss: {loss.item()}')
                self.losses.append(loss.item())
            
    def train(self):
        for num in range(self.epoch):
            self._train_epoch(epoch_num=num)
        return self.losses
    
    def plot_iter_loss(self):
        plt.plot(self.losses)
        plt.title('ConvNet')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()
            
    def save_model(self, path):
        state = {'net_dict': self.net.state_dict()}
        torch.save(state, path)
        logger.info(f'Model saved - {path}')
