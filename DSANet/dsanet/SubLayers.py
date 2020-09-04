""" Define the sublayers in encoder/decoder layer """
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dsanet.Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.n_head = hparams.n_head
        self.d_k = hparams.d_k
        self.d_v = hparams.d_v
        self.drop_prob = hparams.drop_prob
        self.w_qs = nn.Linear(hparams.d_model, hparams.n_head * hparams.d_k)
        self.w_ks = nn.Linear(hparams.d_model, hparams.n_head * hparams.d_k)
        self.w_vs = nn.Linear(hparams.d_model, hparams.n_head * hparams.d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (hparams.d_model + hparams.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (hparams.d_model + hparams.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (hparams.d_model + hparams.d_v)))

        self.attention = ScaledDotProductAttention(hparams)
        self.layer_norm = nn.LayerNorm(hparams.d_model)

        self.fc = nn.Linear(hparams.n_head * hparams.d_v, hparams.d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(hparams.drop_prob)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        if self.hparams.mcdropout == 'True':
            output = nn.functional.dropout(self.fc(output), p=self.drop_prob, training=True)
        else:
            output = self.dropout(self.fc(output))

        #output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """
    #d_model, d_inner, dropout=dropout
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.w_1 = nn.Conv1d(hparams.d_model, hparams.d_inner, 1)
        self.w_2 = nn.Conv1d(hparams.d_inner, hparams.d_model, 1)
        self.layer_norm = nn.LayerNorm(hparams.d_model)
        self.dropout = nn.Dropout(hparams.drop_prob)
        self.drop_prob = hparams.drop_prob

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        if self.hparams.mcdropout == 'True':
            output = nn.functional.dropout(output, p=self.drop_prob, training=True)
        else:
            output = self.dropout(output)
        #output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
