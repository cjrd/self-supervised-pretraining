#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]
    obj = torch.load(input, map_location="cpu")
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        if k[:7] == "module.":
            k=k[7:]
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = v.detach().numpy()

    res = {
        "model": newmodel,
        "__author__": "OpenSelfSup",
        "matching_heuristics": True
    }

    assert sys.argv[2].endswith('.pth')
    with open(sys.argv[2], "wb") as f:
        torch.save(res, f)
