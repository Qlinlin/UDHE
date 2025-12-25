#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /home/zhp/data/zhp/
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from UDHE_model import UDHE

from argparse import ArgumentParser
from myutils.dataloader import UIEBD_Dataset

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of UDHE-UIE')

    # Data parameters

    parser.add_argument('-d', '--dataset-name', help='name of dataset', choices=['UFO', 'UIEB', 'LSUI'],
                        default='UIEB')
    parser.add_argument('-t', '--train-dir', help='training set path', default='your/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='your/val/')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../UDHE_ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--ckpt-load-path', help='start training with a pretrained model', default=None)
    parser.add_argument('--report-interval', help='batch report interval', default=1, type=int)
    parser.add_argument('-ts', '--train-size', nargs='+', help='size of train dataset', default=[256, 256], type=int)
    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=2, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=200, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    """Train UDHE."""

    # Parse training parameters  
    params = parse_args()

    # --- Load training data and validation/test data --- # params.train_size
    train_set = UIEBD_Dataset(params.train_dir, 256, 'train')
    train_loader = DataLoader(train_set, batch_size=params.batch_size,
                              shuffle=True, num_workers=0)
    val_set = UIEBD_Dataset(params.valid_dir, 256, "val")
    valid_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    udhe = UDHE(params, trainable=True)
    udhe.train(train_loader, valid_loader)

