from __future__ import print_function
from __future__ import division

import json
import argparse
from barbar import Bar
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import copy
import torch.utils.data.dataloader

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None

    if model_name == "alexnet":
        """ Alexnet """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "vgg-16":
        """ VGG-16 """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    elif model_name == "mobilenet":
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftr = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftr, num_classes)
        model_ft.num_classes = num_classes
        input_size = 256
    elif model_name == "shufflenet":
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.num_classes = num_classes
        input_size = 256
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    """
    Sets requires_grad to False if we want to reuse the pretrained weights
    :param model:
    :param feature_extracting: whether to 'freeze' the first layers
    :return:
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(rank, model, dataloaders, criterion, optimizer, num_epochs=25):
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
            for indx, (inputs, labels) in enumerate(Bar(dataloaders[phase])):
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
    os.environ['MASTER_ADDR'] = '192.168.2.1'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("mpi", rank=rank, world_size=world_size, group_name='test')

def cleanup():
    dist.destroy_process_group()

def get_num_classes(data_dir):
    num_classes = 0
    for _, dirnames, _ in os.walk(os.path.join(data_dir, "train")):
        num_classes += len(dirnames)

    return num_classes

def preprocess_data(data_dir, batch_size):
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
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x
        in ['train', 'val']}

    print(image_datasets['train'].class_to_idx)
    with open('class_to_idx_map.json', 'w') as fp:
        json.dump(image_datasets['train'].class_to_idx, fp)

    return dataloaders_dict

if __name__ == '__main__':
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',required=True,type=str,help='path to ImageNet root')
    parser.add_argument('--model_name',type=str,default='alexnet',help='pretrained model to use')
    parser.add_argument('--num_epochs', type=int,default=100,help='number of epochs')
    parser.add_argument('--batch_size', type=int,default=1,help='batch size')
    parser.add_argument('--complete_train', action='store_true',help='If set trains the entire network otherwise only the classification layers')
    parser.add_argument('--use_pretrained', type=bool, help='If set uses the pretrained networks from pytorch', default=True)
    parser.add_argument('--rank', type=int, default=0,help='rank of the process')
    parser.add_argument('--world_size',type=int,default=1, help='worldsize for dist')
    parser.add_argument('--distributed', action='store_true')

    args = parser.parse_args()

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    if args.distributed:
        setup(args.rank,args.world_size)

    feature_extracting = not args.complete_train

    model_ft, input_size = initialize_model(args.model_name, get_num_classes(args.image_path), feature_extracting, use_pretrained=True)

    dataloaders_dict = preprocess_data(args.image_path, args.batch_size)

    device = torch.device("cpu")
    model_ft = model_ft.to(device)
    ddp_model = DDP(model_ft, find_unused_parameters=True) if args.distributed else model_ft

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extracting:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=0.001)
    print(optimizer_ft.state_dict())

    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(args.rank, ddp_model, dataloaders_dict, criterion, optimizer_ft, num_epochs=args.num_epochs)
    print('saving model')
    torch.save(model_ft, 'trainedModel.pth')
