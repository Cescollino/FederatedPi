# this file will be used to train and
# report accuracy results of the model

import os
import copy
import time
import pickle

import numpy as np #numbers
from tqdm import tqdm #progress bar
import torch #pytorch
from tensorboardX import SummaryWriter #display metrics
from options import args_parser #parsing arguments when calling py from terminal
from update import LocalUpdate, test_inference

from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details

from torch import nn
class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(3, 5)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(5, 2)
        # Define sigmoid activation and softmax output
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.ReLU(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

model = Network()
model
