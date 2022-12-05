import torch
import torch.nn as nn
import torch.optim as optim

"""
Check if gpu available
"""
def find_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device