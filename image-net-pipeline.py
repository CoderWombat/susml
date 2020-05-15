from __future__ import print_function
from __future__ import division

import json
import argparse
from barbar import Bar
import torch
import torch.nn as nn
import torch.optim as optim
#import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#import matplotlib.pyplot as plt
import time
import os
import copy

def initialize_model(model_ft,num_classes, feature_extract, use_pretrained=True):

    # model_ft = models.vgg19(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(rank,model, dataloaders, criterion, optimizer, num_epochs=25):
    val_acc_history = []
    if rank == 0:
        since = time.time()



        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for indx,(inputs, labels) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc and rank == 0:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
    if rank ==0:
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
    return model, val_acc_history

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.2.108'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("mpi", rank=rank, world_size=world_size,group_name='test')
    #torch.distributed.is_gloo_available()
    #torch.distributed.is_mpi_available()
    #torch.distributed.is_nccl_available()

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',required=True,type=str,help='path to ImageNet root')
    parser.add_argument('--vgg_version',type=str,default='vgg19',help='version of vgg to use')
    parser.add_argument('--num_epochs', type=int,default=100,help='number of epochs')
    parser.add_argument('--batch_size', type=int,default=1,help='batch size')
    parser.add_argument('--complete_train', action='store_true',help='If set trains the entire network otherwise only the classification layers')
    parser.add_argument('--batch_normalization', action='store_true',help='If set uses the vgg model with batch normalization')
    parser.add_argument('--use_pretrained', action='store_true',help='If set uses the pretrained networks from pytorch')
    parser.add_argument('--rank', type=int, default=0,help='rank of the prozess')
    parser.add_argument('--world_size',type=int,default=1, help='worldsize for dist')

    args = parser.parse_args()

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    data_dir = args.image_path
    setup(args.rank,args.world_size)
    if(args.batch_normalization):
        if(args.use_pretrained):
            model = getattr(models, args.vgg_version + '_bn')(pretrained=True)
        else:
            model = getattr(models, args.vgg_version + '_bn')()
    else:
        if (args.use_pretrained):
            model = getattr(models, args.vgg_version)(pretrained=True)
        else:
            model = getattr(models, args.vgg_version)()
    num_classes = 0
    for _, dirnames, _ in os.walk(os.path.join(data_dir,"train")):
        # ^ this idiom means "we won't be using this value"
        num_classes += len(dirnames)
    print(num_classes)

    batch_size = args.batch_size

    num_epochs = args.num_epochs

    feature_extracting = not args.complete_train

    model_ft, input_size = initialize_model(model,num_classes, feature_extracting, use_pretrained=True)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    print(image_datasets['train'].class_to_idx)
    with open('class_to_idx_map.json', 'w') as fp:
        json.dump(image_datasets['train'].class_to_idx, fp)
    # Detect if we have a GPU available
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    ddp_model = DDP(model_ft)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extracting:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=0.001)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(args.rank,ddp_model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
    print('saving model')
    torch.save(model_ft, 'trainedModel.pth')
