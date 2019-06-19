# Imports packages 

import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
from collections import OrderedDict
from PIL import Image

def load_transform(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

                                      



    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    classes = train_datasets.classes
    
    return trainloader, testloader, validloader, classes
    
def load_classes(class_file=None):
    with open(class_file, 'r') as f:
        classes = json.load(f)
    return classes

def network(pretrained_model, hidden_units, output_classes, lr, gpu = False):
    
    if pretrained_model == 'densenet121':
        model = models.densenet121(pretrained = True)
        
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=.3)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p=.3)),
                          ('fc3', nn.Linear(256, output_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        #define the loss function    
        criterion = nn.NLLLoss()
        #define the optimizer with hyperparemeters
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        #use GPU if it is available
        
        
    if pretrained_model == 'vgg16':
        model = models.vgg16(pretrained = True)
        
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=.3)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p=.3)),
                          ('fc3', nn.Linear(256, output_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        #define the loss function    
        criterion = nn.NLLLoss()
        #define the optimizer with hyperparemeters
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        #use GPU if it is available
    if gpu is True:
        model.to('cuda')
        
    return model, optimizer
        
def train(trainloader, validloader, model, optimizer, epochs, print_every, gpu=False):
    running_loss = 0 
    steps = 0
    criterion = nn.NLLLoss()
    #use GPU if it is available
    if gpu is True:
        model.to('cuda')
    #save the train losses and valid losses for every epoch to plot them for test overfitting    
    train_losses=[]    
    valid_losses=[]

    for epoch in range(epochs):
           # Model in training mode, dropout is on
        model.train()
        
        #tracking the training loss for every epoch using train_loss 
        train_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            
            if gpu is True:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            
            if steps % print_every == 0:
                print("the average training loss in the {} epochs from {} epochs is: {:.3f}..".format(epoch+1, epochs,                                             running_loss/print_every))
                train_loss += running_loss
                running_loss = 0
                
            
            #check the training loss and validation loss with the accuracy every epoch
            #set the model in evaluation mode
        model.eval()
           #turn off gradients
        with torch.no_grad():
            valid_loss = 0
            accuracy = 0
            for inputs, labels in validloader:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
            
                #the validation loss for all valid-dataset in every epoch
                valid_loss += batch_loss.item() #convert the loss from tensor to float number
            
                #calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
        train_losses.append(running_loss/len(trainloader))
        valid_losses.append(valid_loss/len(validloader))
        
        print(f"Epoch {epoch+1}/{epochs}.. \n "
             f"The average training loss: {train_loss/len(trainloader):.3f}.. \n"
             f"The average validation loss: {valid_loss/len(validloader):.3f}.. \n"
             f"The average validation accuracy: {accuracy/len(validloader)*100:.3f}")
            # Model in training mode, dropout is on
        model.train()
    # see if there is any overfitting
    #plt.plot(train_losses, label='Training loss')
    #plt.plot(valid_losses, label='Validation loss')
    #plt.legend()    
    

def test(model, testloader, gpu=False):
    if gpu is True:
        model.to('cuda')
    model.eval()
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            if gpu is True:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = torch.exp(model(inputs))
            top_p, top_class = outputs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    model.train()
    print('Accuracy of the network on the test images: %.2f %%' % (100 * accuracy / len(testloader)))
        
def save_checkpoint(state, filename):
    torch.save(state, filename)
    
def load_checkpoint(checkpoint_file, gpu):
    checkpoint = torch.load(checkpoint_file)
    model, optimizer = network(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['output_classes'], checkpoint['learning_rate'], gpu = gpu)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, checkpoint['class_labels']
    
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    
    imageloader = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    im = Image.open(image)
    im = imageloader(im).float()
    img_np = np.array(im)         
    return img_np 
    ''' 
    im = Image.open(image)
    pil_image = im.thumbnail((1000000,256))
    pil_image = im.crop((0,0,224,224))
    np_image = np.array(pil_image)
    np_image = np_image/255
    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    image_normalized = (np_image-image_mean)/image_std
    return image_normalized

#show the image after processing
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax    
       
    
    
def predict(image_path, model, classes, topk, cat, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #no need to work with GPU, it is only one image to predict
    model.to('cpu')
  
    image = process_image(image_path) #process the image to be suitable for the model
    image = torch.from_numpy(image).type(torch.FloatTensor) #convert the numpy image to be tensor
    image.resize_([1,3,224,224]) #resize the image to be in the suitable format for the model(batchsize,rgb,width,height)
    
    #if gpu is True:
        #print('using GPU for prediction')
        #model.to('cuda')
        #image.to('cuda')
        
    logps = model.forward(image) #get the log softmax values from the model
    probs = torch.exp(logps)
    top_p, index = probs.topk(topk)
    top_p = top_p.detach().numpy().tolist()[0]
    index = index.detach().numpy().tolist()[0]
    
    #convert the indexes to classes
    labels = []
    label_index = []
    for i in index:
        label_index.append(int(classes[int(i)]))
    for i in label_index:
        labels.append(cat[str(i)])
        
    return labels, top_p, label_index    
        
        
    
def plot_flower_classes(image_path, model):
    
    plt.figure(figsize = (6,10))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    flower_index = image_path.split('/')[2]
    title_ = cat_to_name[flower_index]

    img = process_image(image_path)
    
    imshow(img, ax=ax1, title=title_);
    
    # Plot bar chart
    
    classes, probs, indexes = predict(image_path, model, topk = 5)
    y_pos = np.arange(len(classes))
    ax2.barh( y_pos , probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel(probs)
    ax2.set_title('predictions of flower names')
    plt.show()    
    
    
   