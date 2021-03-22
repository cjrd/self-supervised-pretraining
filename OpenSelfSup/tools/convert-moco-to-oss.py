#!/usr/bin/env python

import pickle as pkl
import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        old_k = k
        if k.startswith("module.encoder_q.fc."):
            continue
        elif k.startswith("module.encoder_q."):
            k = k.replace("encoder_q.", "")
        else:
            print("Skipping {}".format(old_k))
            continue
        
        # if "layer" not in k:
        #     k = "stem." + k
        # for t in [1, 2, 3, 4]:
        #     k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        # for t in [1, 2, 3]:
        #     k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        # k = k.replace("downsample.0", "shortcut")
        # k = k.replace("downsample.1", "shortcut.norm")
        # print(old_k, "->", k)
        newmodel[k] = v

    res = {"state_dict": newmodel}

    with open(sys.argv[2], "wb") as f:
        torch.save(res, f)
