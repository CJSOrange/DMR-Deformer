''' Define the sublayers in Deformer'''
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Local_Attention(nn.Module):
    ''' Multi-Head local-Attention module '''

    def __init__(self, n_head, n_point, d_model):
        super().__init__()

        self.n_head = n_head
        self.n_point = n_point
        # one linear layer to obtain displacement basis
        self.sampling_offsets = nn.Linear(d_model, n_head * n_point * 3)
        # one linear layer to obtain weight
        self.attention_weights = nn.Linear(2 * d_model, n_head * n_point)
        # self.attention_weights = nn.Linear(2 * d_model, n_head * 3)

    def forward(self, q, k):
        v = torch.cat([q, k], dim=-1)
        n_head, n_point = self.n_head, self.n_point
        sz_b, len_q, len_k = q.size(0), q.size(1), k.size(1)
        # left branch (only moving image)
        sampling_offsets = self.sampling_offsets(q).view(sz_b, len_q, n_head, n_point, 3)
        # right branch (concat moving and fixed image)
        attn = self.attention_weights(v).view(sz_b, len_q, n_head, n_point, 1)
        # attn = self.attention_weights(v).view(sz_b, len_q, n_head, 3)
        # flow = attn
        # attn = F.softmax(attn, dim=-2)
        # multiple and head-wise average
        flow = torch.matmul(sampling_offsets.transpose(3, 4), attn)
        flow = torch.squeeze(flow, dim=-1)
        # sz_b, len_q, 3
        return torch.mean(flow, dim=-2)


class Deformer_layer(nn.Module):
    ''' Compose layers '''

    def __init__(self, d_model, n_head, n_point):
        super(Deformer_layer, self).__init__()
        self.slf_attn = Local_Attention(n_head, n_point, d_model)

    def forward(self, enc_input, enc_input1):
        enc_output = self.slf_attn(enc_input, enc_input1)
        return enc_output


class Deformer(nn.Module):
    '''
    A encoder model with deformer mechanism.
    :param n_layers: the number of layer.
    :param d_model: the channel of input image [batch,N,d_model].
    :param n_position: input image [batch,N,d_model], n_position=N.
    :param n_head: the number of head.
    :param n_point: the number of displacement base.
    :param src_seq: moving seq [batch,N,d_model]
    :param tgt_seq: fixed seq [batch,N,d_model].
    :return enc_output: sub flow field [batch,N,3].
    '''

    def __init__(
            self, n_layers, d_model, n_position, n_head, n_point, dropout=0.1, scale_emb=False):

        super().__init__()

        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            Deformer_layer(d_model, n_head, n_point)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, tgt_seq):

        # -- Forward
        if self.scale_emb:
            src_seq *= self.d_model ** 0.5
            tgt_seq *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)
        enc_output1 = self.dropout(self.position_enc(tgt_seq))
        enc_output1 = self.layer_norm(enc_output1)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, enc_output1)

        return enc_output


if __name__ == '__main__':
    x = torch.rand(3, 112 * 96 * 80, 16)
    y = torch.rand(3, 112 * 96 * 80, 16)
    b, n, d = x.size()
    enc = Deformer(n_layers=1, d_model=d, n_position=n, n_head=8, n_point=64)
    z = enc(x, y)
    print(z.size())
    print(torch.min(z))
