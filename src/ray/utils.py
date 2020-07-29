import torch
import torch.nn as nn
import os

from torchvision import transforms, datasets, models


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

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def get_num_classes(data_dir):
    return 200

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
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, sampler=torch.utils.data.DistributedSampler(image_datasets[x], num_replicas=world_size, rank=rank), num_workers=0) for x
        in ['train', 'val']}

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