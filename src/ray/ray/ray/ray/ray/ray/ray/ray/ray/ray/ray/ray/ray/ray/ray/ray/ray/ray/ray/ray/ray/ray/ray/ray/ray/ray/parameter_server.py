#!/usr/bin/env python
import os
import random
import time

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

from data_worker import DataWorker
from utils import get_num_classes, set_parameter_requires_grad, create_quantized_resnet, \
    create_quantized_mobilenet, create_quantized_shufflenet

random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True

@ray.remote
class ParameterServer(object):
    def __init__(self, args):
        self.args = args

        self.workers = [DataWorker.remote(args, i) for i in range(args.num_workers)]
        print(args.model_name)
        print(get_num_classes(args.image_path))
        self.model, input_size, self.quant_model = self.initialize_model(args.model_name, get_num_classes(args.image_path))

        self.dataloaders_dict = self.preprocess_data(args.image_path, args.batch_size, input_size)

        print("Params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)

        self.optimizer = optim.Adam(params_to_update, lr=0.001)

        self.criterion = nn.CrossEntropyLoss()

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def train(self):
        # model, train_iterators[0], optimizer, criterion, TAG_PAD_IDX, rank, epoch
        # epoch_loss = 0
        # epoch_acc = 0

        self.model.train()

        current_weights = self.get_weights()

        for batch_idx, batch in enumerate(self.dataloaders_dict['train']):
            # import pdb;pdb.set_trace()
            # print('beginning new batch')
            gradients = [
                worker.compute_gradients.remote(current_weights) for worker in self.workers
            ]
            # print('gathering gradients...')
            current_weights = self.apply_gradients(*ray.get(gradients))
            # print(f'weights after batch {i}: {ray.get(current_weights).keys()}')

        # return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def categorical_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
        non_pad_elements = (y != self.TAG_PAD_IDX).nonzero()
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

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

    def evaluate(self):

        # tag_pad_idx

        epoch_loss = 0
        epoch_acc = 0

        # TODO: set model.train() somewhere else before applying workers' gradients!?
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloaders_dict['val']:
                text = batch.text
                tags = batch.udtags
                predictions = self.model(text)
                predictions = predictions.view(-1, predictions.shape[-1])
                tags = tags.view(-1)
                loss = self.criterion(predictions, tags)
                acc = self.categorical_accuracy(predictions, tags)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(self.dataloaders_dict['val']), epoch_acc / len(self.dataloaders_dict['val'])


    def run(self):
        overall_start_time = time.time()

        for epoch in range(self.args.num_epochs):
            print(f'Starting epoch {epoch+1:02}')
            start_time = time.time()
            # train_loss, train_acc = train()
            self.train()
            valid_loss, valid_acc = self.evaluate()
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        overall_end_time = time.time()
        print('took overall', self.epoch_time(overall_start_time, overall_end_time))

        return 1

    def run_async(self):
        overall_start_time = time.time()

        current_weights = self.get_weights()

        updates = len(self.dataloaders_dict['train']) * len(self.workers)
        for epoch in range(self.args.num_epochs):
            gradients = {}
            for worker in self.workers:
                gradients[worker.compute_gradients.remote(current_weights)] = worker

            batches_processed_by_worker = {worker_id: 0 for worker_id in range(self.args.num_workers)}
            start_time = time.time()

            for iteration in range(updates):
                print(f'Starting update {iteration+1:03}/{updates}')
                # train_loss, train_acc = train()
                ready_gradient_list, rest = ray.wait(list(gradients))
                if len(ready_gradient_list) == 0:
                    print(f'wait failed {ready_gradient_list}, {rest}')
                ready_gradient_id = ready_gradient_list[0]
                worker = gradients.pop(ready_gradient_id)
                worker_rank = ray.get(worker.get_rank.remote())
                batches_processed_by_worker[worker_rank] += 1
                self.model.train()
                current_weights = self.apply_gradients(*[ray.get(ready_gradient_id)])

                if batches_processed_by_worker[worker_rank] <= len(self.dataloaders_dict['val']):
                    gradients[worker.compute_gradients.remote(current_weights)] = worker

                # print(f'Update: {iteration+1:02} | Update Time: {epoch_mins}m {epoch_secs}s')

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            valid_loss, valid_acc = self.evaluate()
            # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'Finished epoch {epoch+1:02} in {epoch_mins} min {epoch_secs} s')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        overall_end_time = time.time()
        valid_loss, valid_acc = self.evaluate()
        print(f'Final Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        print('took overall', self.epoch_time(overall_start_time, overall_end_time))

        return 1
