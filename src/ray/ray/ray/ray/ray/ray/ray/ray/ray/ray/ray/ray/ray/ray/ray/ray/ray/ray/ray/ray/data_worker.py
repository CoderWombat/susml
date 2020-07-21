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
            batch = next(self.train_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.train_iterator = iter(self.train_iterator)
            batch = next(self.train_iterator)

        before = datetime.now()
        text = batch.text
        tags = batch.udtags
        # TODO: ?
        # optimizer.zero_grad()
        self.model.zero_grad()
        predictions = self.model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        loss = self.criterion(predictions, tags)
        # self.epoch_acc += categorical_accuracy(predictions, tags, tag_pad_idx).item()
        # acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        loss.backward()
        # self.epoch_loss += loss.item()
        # print(f'finished computing gradients on node {self.rank}')
        print(f'computed gradients for a batch on node {self.rank}, took {datetime.now() - before}...')

        return get_gradients(self.model)
