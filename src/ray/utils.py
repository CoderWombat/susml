import torch
import torch.nn as nn
import os

from torchvision import transforms, datasets, models

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

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def get_num_classes(data_dir):
    num_classes = 0
    for _, dirnames, _ in os.walk(os.path.join(data_dir, "train")):
        num_classes += len(dirnames)

    return num_classes

def preprocess_data(data_dir, batch_size, input_size, world_size, rank):
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
        "train": torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, sampler=torch.utils.data.DistributedSampler(image_datasets['train'], num_replicas=world_size, rank=rank), num_workers=0),
        "val": torch.utils.data.DataLoader(image_datasets['val'],batch_size=batch_size,num_workers=0)
    }
    return dataloaders_dict

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
    elif model_name == "resnet":
        """ Resnet18 """
        model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
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
    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size, quant_model

def get_weights(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}

def set_weights(model, weights):
    model.load_state_dict(weights)

def set_gradients(model, gradients):
    params_to_update = []
    for p in model.parameters():
        if p.requires_grad:
            params_to_update.append(p)

    for g, p in zip(gradients, params_to_update):
        if g is not None:
            p.grad = torch.from_numpy(g)

def get_gradients(model):
    grads = []
    for p in model.parameters():
        if p.requires_grad:
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
    return grads