import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from transformers import *
from model.layers.BiLSTM import BiLSTM_Encoder
from model.layers.Linear import Linear


class Encoder(nn.Module):
    def __init__(self, classes_num, hidden_size=128, BERT_out_dropout=0.5, BiLSTM_dropout=0.1):
        super(Encoder, self).__init__()
        self.BERT = BertModel.from_pretrained('../Pre_trained_BERT/',
                                              output_hidden_states=True,
                                              output_attentions=False)

        self.dropout = nn.Dropout(BERT_out_dropout)

        self.BiLSTM = BiLSTM_Encoder(D=768, hidden_size=hidden_size, dropout=BiLSTM_dropout)

        self.linear1 = Linear(2*hidden_size, 300)
        self.linear2 = Linear(300, 1)
        self.linear3 = Linear(300, classes_num)

    # @torchsnooper.snoop()
    def forward(self, x, mask):

        N, S = x.size()

        last_hidden_states, _, all_hidden_states = self.BERT(x, attention_mask=mask)

        last_hidden_states = self.dropout(last_hidden_states)

        encoder_states, final_state = self.BiLSTM(last_hidden_states, mask)

        mask = mask.view(N, S, 1)

        intermediate = F.gelu(self.linear1(final_state))
        binary_prob = T.sigmoid(self.linear2(intermediate))
        classes_prob = self.linear3(intermediate)

        return binary_prob, classes_prob
