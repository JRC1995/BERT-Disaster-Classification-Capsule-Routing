import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from transformers import *
from model.layers.BiLSTM import BiLSTM_Encoder
from model.layers.Linear import Linear
from model.layers.capsule_net import capsule_fusion

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)


class Encoder(nn.Module):
    def __init__(self, classes_num, hidden_size=128, BERT_out_dropout=0.5, BiLSTM_dropout=0.1):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.zeros = T.tensor(0.0).to(device)
        self.neg_inf = T.tensor(-2.0**32).to(device)

        self.BERT = BertModel.from_pretrained('../Pre_trained_BERT/',
                                              output_hidden_states=True,
                                              output_attentions=False)

        self.dropout = nn.Dropout(BERT_out_dropout)

        self.layer_fusion = capsule_fusion(D=768,
                                           n_in=6, n_out=768,
                                           in_dim=128, out_dim=1,
                                           depth_encoding=True)

        self.hidden_fusion = capsule_fusion(D=768,
                                            n_in=1, n_out=2*hidden_size,
                                            in_dim=128, out_dim=1,
                                            depth_encoding=False)

        self.linear1 = Linear(2*hidden_size, 300)
        self.linear2 = Linear(300, 1)
        self.linear3 = Linear(300, classes_num)
        # @torchsnooper.snoop()

    def forward(self, x, mask):

        N, S = x.size()

        last_hidden_states, _, all_hidden_states = self.BERT(x, attention_mask=mask)

        last_six_hidden_states = all_hidden_states[-6:]
        concated_layers = T.cat(last_six_hidden_states, dim=-1)
        concated_layers = concated_layers.view(N*S, 6, 768)

        _, fused_layers = self.layer_fusion(concated_layers)

        fused_layers = fused_layers.view(N*S, 768)
        fused_layers = fused_layers.view(N, S, 768)

        fused_layers = self.dropout(fused_layers)

        _, fused_hidden = self.hidden_fusion(fused_layers)

        fused_hidden = fused_hidden.view(N, 2*self.hidden_size)

        intermediate = F.gelu(self.linear1(fused_hidden))
        binary_prob = T.sigmoid(self.linear2(intermediate))
        classes_prob = self.linear3(intermediate)

        return binary_prob, classes_prob
