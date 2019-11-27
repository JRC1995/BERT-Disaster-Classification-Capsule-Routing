import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math


class Linear(nn.Module):
    def __init__(self, fan_in, fan_out):
        super(Linear, self).__init__()

        self.linear = nn.Linear(fan_in, fan_out)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)
