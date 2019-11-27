import numpy as np
import math
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = T.device('cuda' if T.cuda.is_available() else 'cpu')

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)


def cross_entropy(model, logits, labels, binary_logits, binary_labels, label_weights, binary_label_weights, mask, gamma1=0.5, gamma2=0.5, l2=1e-6):

    N = binary_labels.size()[0]

    regularization = T.tensor(0.).to(device)  # .to(device)
    blacklist = ["embedding", "bias"]
    for name, param in model.named_parameters():

        if param.requires_grad:
            flag = 0
            for b in blacklist:
                if b in name.lower():
                    flag = 1
                    break

            if flag == 0:
                regularization += T.norm(param).pow(2)

    ce_loss = nn.CrossEntropyLoss(weight=label_weights, reduction='none')
    bce_loss = nn.BCELoss(reduction='mean')

    cross_entropy = ce_loss(logits, labels)
    cross_entropy = cross_entropy.view(N)

    bce = bce_loss(binary_logits.view(N), binary_labels.float())

    masked_cross_entropy = cross_entropy*mask

    masked_cross_entropy = T.mean(masked_cross_entropy)

    if model.training:
        loss = gamma1*bce + gamma2*masked_cross_entropy + l2*regularization
    else:
        loss = gamma1*bce + gamma2*masked_cross_entropy

    return loss
