#!/usr/bin/env python
import argparse
import random
import time

import ray
import torch

from parameter_server import ParameterServer

random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True

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
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--distributed', action='store_true')

    return parser.parse_args()

def run(args):
    ray.init(
        address='auto',
        ignore_reinit_error=True,
        webui_host='0.0.0.0',
        redis_password='5241590000000000'
    )
    try:
        ps = ParameterServer.remote(args)
        # https://docs.ray.io/en/releases-0.8.6/auto_examples/plot_parameter_server.html
        # synchronous parameter server:
        # val = ps.run.remote()
        # asynchronous paramter server:
        val = ps.run_async.remote()
        print(ray.get(val))
    except Exception as e:
        raise e
    finally:
        print('waiting 10s to allow logs to flush')
        time.sleep(10)
        ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    run(args)
