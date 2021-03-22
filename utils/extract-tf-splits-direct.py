#!/usr/local/env python
"""
This utility extract data split ids from tensorflow datasets (v2)

e.g. the command used to grab the resisc45
./extract-tf-splits.py --split=train --outfile tfsplits/resisc45/train.txt --dataid=resisc45 --data=/rscratch/data/tfresisc45/
"""
import argparse
import tensorflow_datasets as tfds 
import tensorflow as tf


parser = argparse.ArgumentParser(description='Extract splits from TF datasets')
parser.add_argument('--data', metavar='DIR', help='path to directory containing the TF compatible data')
parser.add_argument('--split', required=True)
parser.add_argument('--outfile', required=True)
parser.add_argument('--dataid',  required=True)

def main(args):
    ds = tfds.load(args.dataid, split=args.split)
    # Build your input pipeline
    filenames = []
    # ds = ds.take(1)
    for image in ds:
        filenames.append(image['filename'].numpy())


    with open(args.outfile, "wb") as outputf:
        outputf.write(b"\n".join(filenames))

if __name__ == "__main__":
    main(parser.parse_args())