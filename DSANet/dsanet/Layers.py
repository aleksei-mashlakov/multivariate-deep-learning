""" Define the Layers """
import torch.nn as nn
from dsanet.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, hparams):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(hparams)
        self.pos_ffn = PositionwiseFeedForward(hparams)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, hparams):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(hparams)
        self.enc_attn = MultiHeadAttention(hparams)
        self.pos_ffn = PositionwiseFeedForward(hparams)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)

        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn
