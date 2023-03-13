import torch
import torch.nn as nn
import torch.optim as optim

def find_device():
    """ Identifies if GPU is available to run models on, otherwise, uses CPU

    Returns:
        Torch device based on available GPUs or CPUs
    """

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device