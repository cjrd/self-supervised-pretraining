# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################
"""
SVM training using 3-fold cross-validation.

Relevant transfer tasks: Image Classification VOC07 and COCO2014.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import multiprocessing as mp
import tqdm
import argparse
import logging
import numpy as np
import os
import pickle
import sys
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

import svm_helper

import pdb


def task(cls, cost, opts, features, targets):
    out_file, ap_out_file = svm_helper.get_svm_train_output_files(
        cls, cost, opts.output_path)
    if not (os.path.exists(out_file) and os.path.exists(ap_out_file)):
        clf = LinearSVC(
            C=cost,
            class_weight={
                1: 2,
                -1: 1
            },
            intercept_scaling=1.0,
            verbose=0,
            penalty='l2',
            loss='squared_hinge',
            tol=0.0001,
            dual=True,
            max_iter=2000,
        )
        cls_labels = targets[:, cls].astype(dtype=np.int32, copy=True)
        cls_labels[np.where(cls_labels == 0)] = -1
        ap_scores = cross_val_score(
            clf, features, cls_labels, cv=3, scoring='average_precision')
        clf.fit(features, cls_labels)
        np.save(ap_out_file, np.array([ap_scores.mean()]))
        with open(out_file, 'wb') as fwrite:
            pickle.dump(clf, fwrite)
    return 0


def mp_helper(args):
    return task(*args)


def train_svm(opts):
    assert os.path.exists(opts.data_file), "Data file not found. Abort!"
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    features, targets = svm_helper.load_input_data(opts.data_file,
                                                   opts.targets_data_file)
    # normalize the features: N x 9216 (example shape)
    features = svm_helper.normalize_features(features)

    # parse the cost values for training the SVM on
    costs_list = svm_helper.parse_cost_list(opts.costs_list)

    # classes for which SVM training should be done
    if opts.cls_list:
        cls_list = [int(cls) for cls in opts.cls_list.split(",")]
    else:
        num_classes = targets.shape[1]
        cls_list = range(num_classes)

    num_task = len(cls_list) * len(costs_list)
    args_cls = []
    args_cost = []
    for cls in cls_list:
        for cost in costs_list:
            args_cls.append(cls)
            args_cost.append(cost)
    args_opts = [opts] * num_task
    args_features = [features] * num_task
    args_targets = [targets] * num_task

    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm.tqdm(
            pool.imap_unordered(
                mp_helper,
                zip(args_cls, args_cost, args_opts, args_features,
                    args_targets)),
            total=num_task):
        pass


def main():
    parser = argparse.ArgumentParser(description='SVM model training')
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help="Numpy file containing image features")
    parser.add_argument(
        '--targets_data_file',
        type=str,
        default=None,
        help="Numpy file containing image labels")
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help="path where to save the trained SVM models")
    parser.add_argument(
        '--costs_list',
        type=str,
        default="0.01,0.1",
        help="comma separated string containing list of costs")
    parser.add_argument(
        '--random_seed',
        type=int,
        default=100,
        help="random seed for SVM classifier training")

    parser.add_argument(
        '--cls_list',
        type=str,
        default=None,
        help="comma separated string list of classes to train")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    train_svm(opts)


if __name__ == '__main__':
    main()
