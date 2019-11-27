import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math


class BiLSTM_Encoder(nn.Module):
    def __init__(self, D: int, hidden_size: int, dropout: float):
        super(BiLSTM_Encoder, self).__init__()

        self.D = D
        self.hidden_size = hidden_size

        self.initial_hidden_f = nn.Parameter(T.randn(1, hidden_size))
        self.initial_hidden_b = nn.Parameter(T.randn(1, hidden_size))

        self.initial_cell_f = nn.Parameter(T.randn(1, hidden_size))
        self.initial_cell_b = nn.Parameter(T.randn(1, hidden_size))

        self.dropout = nn.Dropout(dropout)

        self.weight_ih = nn.Parameter(T.randn(8, D, hidden_size))
        self.weight_hh = nn.Parameter(T.randn(8, hidden_size, hidden_size))
        self.bias = nn.Parameter(T.zeros(8, 1, hidden_size))

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name.lower():
                nn.init.zeros_(param.data)
            elif 'hidden_state' or 'cell' in name.lower():
                nn.init.zeros_(param.data)
            else:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x, mask):

        N, S, D = x.size()

        mask = mask.view(N, S, 1)

        hidden_f = self.initial_hidden_f
        hidden_b = self.initial_hidden_b

        cell_f = self.initial_cell_f
        cell_b = self.initial_cell_b

        hidden_states_f = []
        hidden_states_b = []

        weight_ih = self.dropout(self.weight_ih)
        weight_hh = self.dropout(self.weight_hh)

        x = x.view(1, N*S, D)

        x_h = T.matmul(x, self.weight_ih)+self.bias
        x_h = x_h.view(8, N, S, self.hidden_size)

        for t in range(S):

            # forward

            hidden_f = hidden_f.view(1, -1, self.hidden_size)

            xf = x_h[0:4, :, t]
            hf = T.matmul(hidden_f, weight_hh[0:4])
            preacts = xf+hf
            gates = T.sigmoid(preacts[0:3])
            f = gates[0]
            i = gates[1]
            o = gates[2]
            cell_ = T.tanh(preacts[3])
            cell_f = f*cell_f + i*cell_
            hidden_f_ = o*T.tanh(cell_f)

            hidden_f = hidden_f.view(-1, self.hidden_size)
            #hidden_f = (1.0-mask[:, t])*hidden_f + mask[:, t]*hidden_f_

            hidden_f = T.where(mask[:, t] == 0.0,
                               hidden_f,
                               hidden_f_)

            hidden_states_f.append(hidden_f.view(1, N, self.hidden_size))

            # backward

            hidden_b = hidden_b.view(1, -1, self.hidden_size)

            xb = x_h[4:, :, S-t-1]
            hb = T.matmul(hidden_b, weight_hh[4:])
            preacts = xb+hb
            gates = T.sigmoid(preacts[0:3])
            f = gates[0]
            i = gates[1]
            o = gates[2]
            cell_ = T.tanh(preacts[3])
            cell_b = f*cell_b + i*cell_
            hidden_b_ = o*T.tanh(cell_b)

            hidden_b = hidden_b.view(-1, self.hidden_size)

            hidden_b = T.where(mask[:, S-t-1] == 0.0,
                               hidden_b,
                               hidden_b_)
            #hidden_b = (1.0-mask[:, S-t-1])*hidden_b + mask[:, S-t-1]*hidden_b_

            hidden_states_b.append(hidden_b.view(1, N, self.hidden_size))

        hidden_states_f = T.cat(hidden_states_f, dim=0)
        hidden_states_f = T.transpose(hidden_states_f, 0, 1)

        hidden_states_b.reverse()
        hidden_states_b = T.cat(hidden_states_b, dim=0)
        hidden_states_b = T.transpose(hidden_states_b, 0, 1)

        hidden_states = T.cat([hidden_states_f, hidden_states_b], dim=-1)*mask
        final_state = T.cat([hidden_states_f[:, -1, :], hidden_states_b[:, 0, :]], dim=-1)

        return hidden_states, final_state
