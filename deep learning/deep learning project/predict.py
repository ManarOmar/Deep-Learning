import matplotlib.pyplot as plt
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

parser.add_argument('--image_path', action ="store", dest="image_path")
parser.add_argument('--checkpoint', action ="store", dest="checkpoint")
parser.add_argument('--top_k', action ="store", default=5, type=int, dest="topk")
parser.add_argument('--category_name', action ="store", default=None, dest="cat")
parser.add_argument('--gpu', action ="store_true", default=False, dest="gpu")

results = parser.parse_args()

model, classes = utilities.load_checkpoint(results.checkpoint, results.gpu)
if results.cat:
    classes_index = utilities.load_classes(results.cat)
labels, probs, label_index = utilities.predict(results.image_path, model, classes,results.topk, classes_index,results.gpu)
print('the label index of the image-the file name-', label_index)
print('the name of the flower :',labels)
print('the probabilities of each name: ',probs)

'''
 python predict.py --image_path 'flowers/test/79/image_06708.jpg' --checkpoint checkpointvgg.pth --category_name cat_to_name.json --top_k 5 --gpu
'''
