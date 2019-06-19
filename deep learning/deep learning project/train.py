import argparse
import sys
import torch
import numpy as np
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
from collections import OrderedDict
from PIL import Image
import utilities

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', action = 'store', dest = 'datadir', help = 'insert the data directory')
parser.add_argument('--class_file', action = 'store', dest = 'class_file', default = None, help = 'insert the class file')
parser.add_argument('--gpu', action = 'store_true', dest = 'gpu', default=False, help = 'use gpu if available')
parser.add_argument('--arch', action = 'store', default='vgg16', dest = 'pre_model', help = 'pretrained model name')
parser.add_argument('--learning_rate', action = 'store', default=.001 ,dest = 'lrate', type=float, help = 'learning rate value for optimization')
parser.add_argument('--hidden_units', action = 'store', default=512, dest = 'hidden_units', type=int, help = 'number of hidden units')
parser.add_argument('--epochs', action = 'store', default=5, dest = 'epochs', type=int, help = 'number of epochs')
parser.add_argument('--output', action = 'store', dest = 'output', type=int, help = 'number of output classes')
parser.add_argument('--print_every', action = 'store', default=20, dest = 'print_every', type=int, help ='print losses and accuracy in every the number of batches specified here')
parser.add_argument('--train', action = 'store_true', dest = 'train', default=False, help = 'set true to start training and false if you do not want')
parser.add_argument('--save_dir', action = 'store', dest = 'save', default = None, help = 'set the checkpoint file name')


results = parser.parse_args()

if results.train == True:
    #load data into image loaders (train, test, validation)
    trainloader, testloader, validloader, classes = utilities.load_transform(results.datadir)
    #load the classes into training process if exist
    
    model, optimizer = utilities.network(results.pre_model, results.hidden_units, results.output, results.lrate, results.gpu)
    utilities.train(trainloader, validloader, model, optimizer, results.epochs, results.print_every, results.gpu)
    utilities.test(model, testloader, results.gpu)
    
    saved_dict = {'arch' : results.pre_model,
                               'hidden_units' : results.hidden_units,
                               'class_labels' : classes,
                               'epochs': results.epochs,
                               'learning_rate': results.lrate,
                               'output_classes' : results.output,
                               'model_state_dict' : model.state_dict(),
                               'optimizer' : optimizer.state_dict()}

    utilities.save_checkpoint(saved_dict, results.save)
    print("the checkpoint was saved in {}".format(results.save))
'''
The average validation accuracy: 87.019
Accuracy of the network on the test images: 83.57 %
'''
'''
python train.py --data_dir flowers --class_file cat_to_name.json --gpu --arch vgg16 --learning_rate .001 --hidden_unit 512 --epochs 5 --output 102 --print_every 20 --train --save_dir checkpointvgg.pth
'''
   


