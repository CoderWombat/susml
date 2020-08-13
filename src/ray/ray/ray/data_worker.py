#!/usr/bin/env python
import os
import random
from datetime import datetime

import ray
import torch
import torch.nn as nn
from torch import optim

from utils import get_num_classes, initialize_model, preprocess_data, get_gradients, set_weights

random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True

@ray.remote(num_cpus=3)
class DataWorker(object):
    def __init__(self, args, rank):
        self.rank = rank
        self.epoch_loss = 0
        self.epoch_acc = 0

        self.args = args

        self.model, input_size, self.quant_model = initialize_model(args.model_name,
                                                                         get_num_classes(args.image_path))

        self.dataloaders_dict = preprocess_data(args.image_path, args.batch_size, input_size)

        self.train_iterator = iter(self.dataloaders_dict['train'])

        print("Params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)

        self.optimizer = optim.Adam(params_to_update, lr=0.001)

        self.criterion = nn.CrossEntropyLoss()


    def get_rank(self):
        return self.rank

    def compute_gradients(self, weights):
        # print(f'computing gradients for a batch on node {self.rank} at {datetime.now()}...')
        set_weights(self.model, weights)

        try:
            (inputs, labels) = next(self.train_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.train_iterator = iter(self.dataloaders_dict['train'])
            (inputs, labels) = next(self.train_iterator)

        before = datetime.now()

        self.model.zero_grad()

        if self.quant_model != None:
            # Use quantized model in inference mode - only train on extracted features
            quant_outputs = self.quant_model(inputs)
            outputs = self.model(quant_outputs)
        else:
            outputs = self.model(inputs)

        loss = self.criterion(outputs, labels)
        loss.backward()

        # print(f'computed gradients for a batch on node {self.rank}, took {datetime.now() - before}...')

        if self.quant_model:
            return get_gradients(self.quant_model)
        else:
            return get_gradients(self.model)
