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
import psutil
import torch.multiprocessing as mp
#import mkl

RESNET_50_QUANTIZED_URL = 'https://s3.amazonaws.com/download.caffe2.ai/models/resnet50_quantized/resnet50_quantized_predict_net.pb'


def create_quantized_resnet(model_fe, num_ftrs, num_classes):
    model_fe_features = nn.Sequential(
        model_fe.quant,  # Quantize the input
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.layer4,
        model_fe.avgpool,
        model_fe.dequant,  # Dequantize the output
    )

    new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes),
    )

    new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
    )

    return new_model


def create_quantized_mobilenet(model_fe, num_ftrs, num_classes):
    class Reshape(nn.Module):
        def __init__(self):
            super(Reshape, self).__init__()

        def forward(self, x):
            return x.reshape(x.shape[0], -1)

    quant_model = nn.Sequential(
        model_fe.quant,  # Quantize the input
        model_fe.features,
        model_fe.dequant,  # Dequantize the output
    )

    new_model = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        Reshape(),
        nn.Linear(num_ftrs, num_classes),
    )

    return new_model, quant_model


def create_quantized_shufflenet(model_fe, num_ftrs, num_classes):
    model_fe_features = nn.Sequential(
        model_fe.quant,  # Quantize the input
        model_fe.conv1,
        model_fe.maxpool,
        model_fe.stage2,
        model_fe.stage3,
        model_fe.stage4,
        model_fe.conv5,
        model_fe.dequant,  # Dequantize the output
    )

    new_model = nn.Sequential(
        model_fe_features,
        nn.Linear(num_ftrs, num_classes)
    )

    return new_model


def initialize_model(model_name, num_classes):
    model = None
    quant_model = None

    if model_name == "alexnet":
        """ Alexnet """
        model = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "vgg-16":
        """ VGG-16 """
        model = models.vgg16_bn(pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        """ Squeezenet """
        model = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        input_size = 224
    elif model_name == "resnet":
        """ Resnet18 """
        model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "quantized_resnet":
        """ Resnet18 - Quantized """
        quant_res = models.quantization.resnet18(pretrained=True, progress=True, quantize=True)
        num_ftrs = quant_res.fc.in_features
        model = create_quantized_resnet(quant_res, num_ftrs, num_classes)
        input_size = 224
    elif model_name == "mobilenet":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        set_parameter_requires_grad(model)
        num_ftr = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftr, num_classes)
        model.num_classes = num_classes
        input_size = 224
    elif model_name == "quantized_mobilenet":
        quant_mob = models.quantization.mobilenet_v2(pretrained=True, progress=True, quantize=True)
        num_ftrs = quant_mob.classifier[1].in_features
        model, quant_model = create_quantized_mobilenet(quant_mob, num_ftrs, num_classes)
        input_size = 224
    elif model_name == "shufflenet":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.num_classes = num_classes
        input_size = 256
    elif model_name == "quantized_shufflenet":
        quant_shuf = models.quantization.shufflenet_v2_x1_0(pretrained=True, progress=True, quantize=True)
        num_ftrs = quant_shuf.fc.in_features
        model = create_quantized_shufflenet(quant_shuf, num_ftrs, num_classes)
        input_size = 256
    elif model_name == "efficientnet":
        model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.num_classes = num_classes
        input_size = 256
    elif model_name == "densenet":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.num_classes = num_classes
        input_size = 256
    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size, quant_model


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def train_model(rank, model, quant_model, dataloaders, criterion, optimizer, num_epochs=25):
    print(f"training started with rank {rank}")
    device = torch.device("cpu")

    if rank == 0:
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                if args.distributed:
                    print(str(rank) + ' blocking')
                    torch.distributed.barrier()
                    print(str(rank) + ' unblocked')
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for indx, (inputs, labels) in enumerate(Bar(dataloaders[phase])):
                #psutil.cpu_percent(interval=None)
                print(psutil.cpu_percent(interval=None, percpu=True))
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if quant_model != None:
                        # Use quantized model in inference mode - only train on extracted features
                        quant_outputs = quant_model(inputs)
                        outputs = model(quant_outputs)
                    else:
                        outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if rank == 0:
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print(str(rank) + 'finished train and val')

    if rank == 0:
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
    return model

def run_inference(i, args):
    print("hi")
    model = args[0]
    inputs = args[1]
    results = args[3]

    results.append((i, model(inputs)))



def cleanup():
    dist.destroy_process_group()


def get_num_classes(data_dir):
    num_classes = 0
    for _, dirnames, _ in os.walk(os.path.join(data_dir, "train")):
        num_classes += len(dirnames)

    return num_classes


def preprocess_data(data_dir, batch_size, input_size):
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val']}

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x
        in ['train', 'val']}

    return dataloaders_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, type=str, help='path to ImageNet root')
    parser.add_argument('--model_name', type=str, default='alexnet', help='pretrained model to use')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    # parser.add_argument('--rank', type=int, default=0, help='rank of the process')
    # parser.add_argument('--world_size', type=int, default=1, help='worldsize for dist')
    # parser.add_argument('--quantized', type=bool, help='Set to true when using quantized model', default=True)
    parser.add_argument('--dataloader_workers', type=int, default=0)
    parser.add_argument('--distributed', action='store_true')
    os.environ["OMP_NUM_THREADS"] = '1'
    print(torch.__config__.parallel_info())

    return parser.parse_args()


def run(rank):
    model, input_size, quant_model = initialize_model(args.model_name, get_num_classes(args.image_path))

    dataloaders_dict = preprocess_data(args.image_path, args.batch_size, input_size)

    device = torch.device("cpu")
    model = model.to(device)

    params_to_update = model.parameters()

    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    ddp_model = DDP(model, find_unused_parameters=True) if args.distributed else model

    optimizer = optim.Adam(params_to_update, lr=0.001)

    criterion = nn.CrossEntropyLoss()

    model = train_model(rank, ddp_model, quant_model, dataloaders_dict, criterion, optimizer,
                              num_epochs=args.num_epochs)

    torch.save(model.state_dict(), "testmodel")


if __name__ == '__main__':
    torch.set_num_threads(4)
    mp.set_start_method("spawn")
    print("possible threads: " + str(torch.get_num_threads()))

    args = parse_args()

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # rank = int(os.environ['OMPI_COMM_WORLD_RANK']) if 'OMPI_COMM_WORLD_RANK' in os.environ else 0
    # world_size= int(os.environ['OMPI_COMM_WORLD_SIZE']) if 'OMPI_COMM_WORLD_SIZE' in os.environ else 0
    # print("process with rank %d started in world size &d" % rank, world_size)

    if args.distributed:
        dist.init_process_group("mpi")
        run(dist.get_rank())
    else:
        run(0)
