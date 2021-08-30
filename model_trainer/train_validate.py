import torch
from model_trainer.neural_net import ConvNet, TrainNeuralNet
from settings import conv_net_pkl, logger

# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class TrainValidateModel:
    def __init__(self):
        pass
    
    @staticmethod
    def train_dataset(dataset):
        # Train the network
        logger.info(f'Initiating model training - {device}')
        trainer = TrainNeuralNet(dataset=dataset, net=ConvNet(), device=device)
        trainer.train()
        # Plot network learning trend
        logger.info('Plotting the model loss vs iterations')
        trainer.plot_iter_loss()
        # Saving the model as a state dictionary
        logger.info('Saving the model')
        trainer.save_model(path=conv_net_pkl)
        
    @staticmethod
    def load_model():
        # Load the trained model
        logger.info(f'Loading the model - {device}')
        net = ConvNet()
        state = torch.load(conv_net_pkl, map_location=device)
        net.load_state_dict(state['net_dict'])
        return net.to(device)
    
    @staticmethod
    def validate_dataset(dataset):
        logger.info(f'Validating the model - {device}')
        net = TrainValidateModel.load_model()
        correct = 0.0
        total = 0.0
        for data in dataset:
            image, label = data
            image = image.to(device)
            label = label.to(device)
            # Doing the Forward pass
            result = net(image)
            # Converting the predictions to probabilities, by applying the softmax function
            result_sm = torch.nn.functional.softmax(result, dim=1)
            # Finding the prediction with the largest probability
            _, pred = torch.max(result_sm.data, 1)
            total += label.size(0)
            # correct is incremented by the number of prediction which are correct (equal to the ground truth labels)
            correct += (pred == label).sum().item()
        logger.info("Accuracy of Test Data: {0:.2f}%".format(correct/total *100))
