#!/usr/bin/env python
import random
import time

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim

from data_worker import DataWorker
from utils import get_num_classes, initialize_model, preprocess_data

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
        self.model, input_size, self.quant_model = initialize_model(args.model_name, get_num_classes(args.image_path))

        self.dataloaders_dict = preprocess_data(args.image_path, args.batch_size, input_size)

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
