#!/usr/bin/env python
"""
This utility computes the mean, var, and std of pixel values of a large image dataset
e.g.
./compute-dataset-pixel-mean-std.py --data /path/to/image-folder

where image-folder has the structure from ImageFolder in pytorch
class/image-name.jp[e]g
or whatever image extension you're using
"""

import sys
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Compute image statistics from ImageFolder')
parser.add_argument('--data', metavar='DIR', help='path to image directory with structure class/image.ext', required=True)
parser.add_argument('--numworkers', type=int, default=30)
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--numbatches', type=int, default=-1)


def main(args):
    dataset = datasets.ImageFolder(args.data , transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x:  torch.stack([x.mean([1,2]), (x*x).mean([1,2])]) )])) # x.view(x.shape[0], -1))]))
    
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        num_workers=args.numworkers,
        shuffle=True
    )

    mean = 0.
    nb_samples = 0.
    results = torch.zeros((2,3))
    N = len(dataset)
    Nproc = 0
    i = 0
    if args.numbatches < 0:
        NB = len(loader)
    else:
        NB = args.numbatches
    for data, _ in loader:
        results += data.sum(0)
        Nproc += data.shape[0]        
        i += 1
        print("batch: {}/{}".format(i, NB))
        if i >= NB:
            break
        
    print(results)
    means = results[0,:] / Nproc
    sqsums = results[1,:] / Nproc
    vvars = sqsums - means**2
    print('means: {}'.format(means))
    print('vars: {}'.format(vvars))
    stds = vvars**0.5
    print('stds: {}'.format(stds))


if __name__ == "__main__":
    main(parser.parse_args())
