import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, hparams): # temperature, attn_dropout=0.1):
        super().__init__()
        self.hparams = hparams
        self.temperature = np.power(hparams.d_k, 0.5)
        self.drop_prob = hparams.drop_prob
        self.dropout = nn.Dropout(hparams.drop_prob)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        if self.hparams.mcdropout == 'True':
            attn = nn.functional.dropout(attn, p=self.drop_prob, training=True)
        else:
            attn = self.dropout(attn)
        #attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
