# CNN training on edge devices

This repo contains code for running different convolutional neural networks on a Raspberry Pi cluster. The use case 
is transfer learning (only classification layers are learned) on a subset of ImageNet.

## Run code

- Start ray

`ray stop && OMP_NUM_THREADS=3 ray start --address='192.168.0.13:6379' --redis-password='5241590000000000'`

- Start training

`python ray/main.py`

- Parameters:

    - _image_path_: Path to dataset (format: data/train/<class_names>/<image_files> + data/val/<class_names>/<image_files>)
    - _model_name_: One of ['alexnet', 'resnet', 'mobilenet', 'quantized_mobilenet']
    - _num_epochs_: Number of epochs
    - _batch_size_: Batch size
    - _num_workers_: How many worker nodes to use
    - _sync_param_server_: Whether to use a synchronous instead of an async parameter server 
