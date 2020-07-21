import torch.nn as nn
import os

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
    num_classes = 0
    for _, dirnames, _ in os.walk(os.path.join(data_dir, "train")):
        num_classes += len(dirnames)

    return num_classes