import random
import numpy as np
import torch


def torch_seed(seed=0):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normal_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def dictConv(orgDict, updatedDict):
    for key1 in updatedDict:
        val = updatedDict[key1]
        if isinstance(val, dict):
            if not key1 in orgDict or orgDict[key1] is None:
                orgDict[key1] = {}
            dictConv(orgDict[key1], val);
        else:
            orgDict[key1] = val;