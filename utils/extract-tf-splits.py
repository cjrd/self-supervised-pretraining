#!/usr/local/env python
"""
This utility extract data split ids from tensorflow datasets from VTAB (using TFv1)

e.g. the command used to grab the resisc45
./extract-tf-splits.py --split=train --outfile tfsplits/resisc45/train.txt --dataid=resisc45 --data=/rscratch/data/tfresisc45/
"""
import argparse
import tensorflow as tf
import task_adaptation.data_loader as data_loader
from tensorflow.compat.v1.data import make_one_shot_iterator
from task_adaptation.registry import Registry


parser = argparse.ArgumentParser(description='Extract splits from TF datasets')
parser.add_argument('--data', metavar='DIR', help='path to directory containing the TF compatible data')
parser.add_argument('--split', required=True)
parser.add_argument('--outfile', required=True)
parser.add_argument('--dataid',  required=True)
def get_data_params_from_flags(args):
  return {
      "dataset": "data.{}".format(args.dataid),
      "dataset_train_split_name": args.split,
      "dataset_eval_split_name": "val",
      "shuffle_buffer_size": 100,
      "prefetch": 100,
      "train_examples": None,
      "batch_size": 32,
      "batch_size_eval": 32,
      "data_for_eval": True,
      "data_dir": args.data,
      "input_range": [0.0, 1.0]
  }

def main(args):
    data_params = get_data_params_from_flags(args)
    rdata = Registry.lookup(data_params["dataset"],
                         kwargs_extra={"data_dir": data_params["data_dir"]})
    nsamples=rdata.get_num_samples(data_params["dataset_train_split_name"]) 
    
    input_fn = data_loader.build_data_pipeline(data_params, mode="train")
    data = make_one_shot_iterator(input_fn({"batch_size": 32})).get_next()
    filenames=[]
    with tf.Session() as sess:
        while True:
            try:
                filenames += sess.run(data)["filename"].tolist()
                print(len(filenames))
                if len(filenames) > (nsamples*2):
                    break
            except:
                break
        with open(args.outfile, "wb") as outputf:
            outputf.write(b"\n".join(list(set(filenames))))
if __name__ == "__main__":
    main(parser.parse_args())