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

        self.hidden_size = hidden_size

        self.BERT = BertModel.from_pretrained('../Pre_trained_BERT/',
                                              output_hidden_states=True,
                                              output_attentions=False)

        self.dropout = nn.Dropout(BERT_out_dropout)

        self.BiLSTM = BiLSTM_Encoder(D=768, hidden_size=hidden_size, dropout=BiLSTM_dropout)

        self.linear_attn_1 = Linear(768, 300)
        self.linear_attn_2 = Linear(300, 1)

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
        concated_layers = concated_layers.view(N*S*6, 768)

        attn_scores = self.linear_attn_2(T.tanh(self.linear_attn_1(concated_layers)))
        attn_scores = attn_scores.view(N*S, 6)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = attn_scores.view(N*S, 6, 1)

        concated_layers = concated_layers.view(N*S, 6, 768)

        fused_layers = T.sum(attn_scores*concated_layers, dim=1)

        fused_layers = fused_layers.view(N, S, 768)

        fused_layers = self.dropout(fused_layers)

        encoder_states, final_state = self.BiLSTM(fused_layers, mask)

        mask = mask.view(N, S, 1)

        # Attention Mechanism

        intermediate = F.gelu(self.linear1(final_state))
        binary_prob = T.sigmoid(self.linear2(intermediate))
        classes_prob = self.linear3(intermediate)

        return binary_prob, classes_prob
