import matplotlib.pyplot as plt
import json
import os
from os.path import basename
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--fname', type=str, default='fig.png',
                            help='name of the plot file.')
parser.add_argument('--folder', type=str, default='../OpenSelfSup/work_dirs/classification/ucmerced/r50_custom_mv_lr0p01_warmup_train_5_val_20_test_20_2',
        help='Path of the folder. e.g.: ../OpenSelfSup/work_dirs/classification/ucmerced/r50_custom_mv_lr0p01_warmup_train_5_val_20_test_20_2')

args = parser.parse_args()


def multi_plot(root):
    root_dir = root
    num = len([fn for fn in os.listdir(root_dir) if ".pth" in fn])
    print(num)

    plt.figure(figsize=(17,60))
    plt_idx = 1
    try:
        for root, _, files in os.walk(root_dir):
            if root.endswith(".pth"):
                print(root)
                for fname in os.listdir(root):
                    if 'train' in fname and '.log' in fname:
                        json_name = os.path.join(root, fname)

                        lines = [line.strip() for line in open(json_name)]
                        idx = 0
                        val_top1, val_top5, test_top1, test_top5 = [], [], [], []
                        
                        top1_lst = []
                        for line in lines:
                            if "- head0_top1" in line:
                                top1_lst.append(float(line.split(":")[-1].strip()))
                        val_top1 = top1_lst[::2]
                        test_top1 = top1_lst[1::2]
                        print(f"=====> length of val_top1 is: {len(val_top1)}")
                        
                        x = range(len(val_top1))
                        
                        plt.subplot(num, 2, plt_idx)
                        
                        plt.title(basename(root)[:-4] + f"\nMax : {test_top1[np.argmax(val_top1)]}")
                        plt.ylabel("accuracy")
                        plt.xlabel("epochs")
                        
                        plt.plot(x, val_top1, '-b', label='val_top1')
                        plt.plot(x, test_top1, '-r', label='test_top1')
                        plt.legend()
                        plt.ylim(0, 100)
                        plt.grid(color='k', linestyle='--', linewidth=0.4)
                        plt_idx += 1
                        
                        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.96, hspace=0.9, wspace=0.3)

    except:
        pass
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.96, hspace=0.9, wspace=0.3)
    plt.savefig(args.fname)
#     plt.tight_layout()
#     plt.suptitle(root_dir)
#     plt.text(10, 10, "===========================")

multi_plot(args.folder)



