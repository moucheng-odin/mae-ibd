# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import re
import time

from collections import OrderedDict
from pathlib import Path

import yaml

from clearml_tools.remote import Trainer


from torch.utils.data import Dataset

import sys
import collections.abc as container_abcs
from collections import namedtuple

SixPatch = namedtuple("six", ["container_abcs"])
_six = SixPatch(container_abcs=container_abcs)
sys.modules["torch._six"] = _six

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

# sys.modules["torch._six.container_abcs"] = container_abcs


import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch

from PIL import Image

from clearml_tools.remote import Trainer

class CustomImageDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        with open(txt_file, 'r') as f:
            self.img_paths = [line.strip() for line in f]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    # add an argument for clearml:
    parser.add_argument("--config-file", type=str, default=config_file, help="config file to use")
    parser.add_argument("--clearml", choices=["local", "remote", "disable"], help="ClearML options for reporting/remote execution")

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    parser.add_argument('--data_path', default='/mount/ssd_sdd/Data/Pollux/data_split/Mayo/all_matching_frames_v4_2024_01_12/manual_split_total/split27', type=str,
                        help='dataset path')
    
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', default=False, help='True for enabling distributed training, false for single GPU')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


# def main(args):
def main(params):
    # with open(config_file, "r") as f:
    #     args = yaml.full_load(f)
    
    print(params)
    
    if params["dir_output"]:
        Path(params["dir_output"]).mkdir(parents=True, exist_ok=True)
    
    # misc.init_distributed_mode(args)
    misc.init_distributed_mode(params)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(params).replace(', ', ',\n'))

    device = torch.device(params["device"])

    # fix the seed for reproducibility
    seed = params["seed"] + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(params["input_size"], scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # the existing dataset train needs a specific data structure, we would like to use another dataset by reading the txt files of images paths:
    txt_train_path = os.path.join(params["data_path"], 'txt_files' ,'0_train_images.txt')
    dataset_train = CustomImageDataset(txt_file=txt_train_path, transform=transform_train)
    
    print(dataset_train)

    # if True:  # args.distributed:
    if params["distributed"] is True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        global_rank = misc.get_world_size()

    if global_rank == 0 and params["log_dir"] is not None:
        os.makedirs(params["log_dir"], exist_ok=True)
        log_writer = SummaryWriter(log_dir=params["log_dir"])
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        pin_memory=params["pin_mem"],
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[params["model"]](norm_pix_loss=params["norm_pix_loss"])

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = params["batch_size"] * params["accum_iter"] * misc.get_world_size()
    # print(type(eff_batch_size))
    # print(type(params["blr"]))
    blr = float(params["blr"])
    
    if params["lr"] is None:  # only base_lr is specified
        # print("lr is not specified, using base_lr")
        params["lr"] = blr * eff_batch_size / 256
        
    # # check the data type;
    # print(params["lr"].dtype)
    # print(eff_batch_size.dtype)

    print("base lr: %.2e" % (params["lr"] * 256 / eff_batch_size))
    print("actual lr: %.2e" % params["lr"])

    print("accumulate grad iterations: %d" % params["accum_iter"])
    print("effective batch size: %d" % eff_batch_size)

    if params["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params["gpu"]], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, params["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups, lr=params["lr"], betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=params, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {params['epochs']} epochs")
    start_time = time.time()
    for epoch in range(params["start_epoch"], params["epochs"]):
        if params["distributed"]:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=params
        )
        if params["dir_output"] and (epoch % 20 == 0 or epoch + 1 == params["epochs"]):
            misc.save_model(
                args=params, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if params["dir_output"] and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(params["dir_output"], "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    
    t1 = time.time()
    
    # modifying it to make it work with clearml:
    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Could not set multiprocessing start method to spawn. Tasks may hang.")


    # parser.add_argument("--config-file", type=str, default=config_file, help="config file to use")
    # parser.add_argument("--output-dir", type=str, default=None, help="output folder to store reproduced results in")
    
    # args = parser.parse_args()
    # config_file = args.config_file
    
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # main(config_file)
    
    # args = get_args_parser()
    # args = args.parse_args()
    
    parser = argparse.ArgumentParser(description="Pre Train a Masked AutoEncoder (MAE) model")
    parser.add_argument("--clearml", choices=["local", "remote", "disable"], help="ClearML options for reporting/remote execution")
    parser.add_argument("--config-file", type=str, help="config file to use")
    parser.add_argument("--output-folder", type=str, default=None, help=" (optional) output folder to store reproduced results in")

    args = parser.parse_args()
    trainer = Trainer(
        func=main,
        args=args,
        config_file=args.config_file,
        root_dir=Path(__file__).absolute().parent.parent.parent,
    )
    
    trainer.train()
    
    # main(args)

    t2 = time.time()
    print(f"\nTraining took {t2 - t1} seconds.")

