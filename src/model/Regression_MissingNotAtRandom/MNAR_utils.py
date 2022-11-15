import torch
import numpy as np
pi = torch.FloatTensor([np.pi])

def pi_star(y):
    return torch.exp(1 - y) / (1 + torch.exp(1 - y))


def dnorm(y):
    return torch.exp(-y ** 2 / 2) / torch.sqrt(2 * pi)


def ddnorm(y):
    return torch.exp(-y ** 2 / 2) / torch.sqrt(2 * pi) * (-y)