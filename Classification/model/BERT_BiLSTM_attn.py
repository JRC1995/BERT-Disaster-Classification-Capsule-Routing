import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from transformers import *
from model.layers.BiLSTM import BiLSTM_Encoder
from model.layers.Linear import Linear

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)


class Encoder(nn.Module):
    def __init__(self, classes_num, hidden_size=128, BERT_out_dropout=0.5, BiLSTM_dropout=0.1):
        super(Encoder, self).__init__()

        self.zeros = T.tensor(0.0).to(device)
        self.neg_inf = T.tensor(-2.0**32).to(device)
        self.hidden_size = hidden_size

        self.BERT = BertModel.from_pretrained('../Pre_trained_BERT/',
                                              output_hidden_states=True,
                                              output_attentions=False)

        self.dropout = nn.Dropout(BERT_out_dropout)

        self.BiLSTM = BiLSTM_Encoder(D=768, hidden_size=hidden_size, dropout=BiLSTM_dropout)

        self.linear_attn_1 = Linear(2*hidden_size, 300)
        self.linear_attn_2 = Linear(300, 1)

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

        # Attention Mechanism

        attention_mask = T.where(mask == 0.0, self.neg_inf, self.zeros)
        attention_mask = attention_mask.view(N, S)

        encoder_states = encoder_states.view(N*S, 2*self.hidden_size)

        attn_scores = self.linear_attn_2(T.tanh(self.linear_attn_1(encoder_states)))
        attn_scores = attn_scores.view(N, S)
        attn_scores = attn_scores+attention_mask
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = attn_scores.view(N, S, 1)

        encoder_states = encoder_states.view(N, S, 2*self.hidden_size)

        context_vector = T.sum(attn_scores*encoder_states, dim=1)

        intermediate = F.gelu(self.linear1(context_vector))
        binary_prob = T.sigmoid(self.linear2(intermediate))
        classes_prob = self.linear3(intermediate)

        return binary_prob, classes_prob
