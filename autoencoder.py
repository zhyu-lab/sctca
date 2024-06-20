import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath
import math

seq_lens = []
kernel_sizes = []
head_dims = [32, 16, 8]
n_layers = 2
n_channels = 128
# seg_len_max = 500  # for simu
seg_len_max = 500  # for chisel


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.05)
            # m.bias.data.zero_()


class WMSA(nn.Module):
    """ Self-attention module in 1D window Transformer
    """
    def __init__(self, input_dim, output_dim, head_dim, window_size):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = nn.Parameter(self.relative_position_params.view(2*window_size-1, self.n_heads).transpose(0, 1))

    def generate_mask(self, w, p, padding_size):
        """ generating the mask of 1D W-MSA
        Args:
            :param w: number of windows
            :param p: window size
            :param padding_size: padding size
        Returns:
            mask: should be (1 1 w p p),
        """
        mask = torch.zeros(w, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        s = p - padding_size
        mask[-1, :s, s:] = True
        mask[-1, s:, :s] = True
        mask = rearrange(mask, 'w p1 p2 -> 1 1 w p1 p2')
        return mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b n c]
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b n c]
        """

        original_n = x.size(1)
        remainder = original_n % self.window_size
        needs_padding = remainder > 0
        padding_size = 0
        if needs_padding:
            padding_size = self.window_size - remainder
            x = F.pad(x, (0, 0, 0, padding_size, 0, 0), value=0)
        x = rearrange(x, 'b (w p) c -> b w p c', p=self.window_size)
        windows = x.size(1)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b w p (threeh c) -> threeh b w p c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        attn_mask = self.generate_mask(windows, self.window_size, padding_size)
        sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b w p c -> b (w p) c', w=windows, p=self.window_size)

        return output[:, :original_n]

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i] for i in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long()]


class WindowBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size):
        """ WindowTransformer Block
        """
        super(WindowBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size)
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, dropout=0):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k = q.size(2)
        scores = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)
        return attn, context


class ConvLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        y = x + self.cnn(x)
        return y


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, d_model, head_dim, window_size, dropout=0):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.attention = SelfAttention(dropout)
        self.window_size = window_size
        self.windowAttention = WindowBlock(in_channels, in_channels, head_dim, window_size)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        seq_len = x.size(1)
        w = int(np.ceil(seq_len / seg_len_max))
        y = x.clone()
        count = 0
        for i in range(w):
            ids = torch.arange(i, seq_len, w)
            count += len(ids)
            sec = x[:, ids]
            attn, r = self.attention(sec, sec, sec)
            y[:, ids] = r
        assert count == seq_len

        y = self.windowAttention(y)
        return torch.transpose(y, 1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, head_dim, window_size, kernel_size=3):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0)
        self.attlayer = AttentionLayer(int(in_channels/2), int(in_channels/2), head_dim, window_size)
        self.cnn = ConvLayer(in_channels-int(in_channels/2))
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=2)
        self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        y = self.conv1(x)
        split_size = int(y.size(1)/2)
        y1, y2 = torch.split(y, [split_size, y.size(1)-split_size], dim=1)
        y1 = self.attlayer(y1)
        y2 = self.cnn(y2)
        y = torch.cat((y1, y2), dim=1)
        y = x + self.conv2(y)
        y = self.conv3(y)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DecoderLayer, self).__init__()
        self.cnn = ConvLayer(in_channels)
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2)

    def forward(self, x):
        y = self.cnn(x)
        y = self.conv1(y)
        return y


class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(int(n_channels/pow(2, i)),
                                                  int(n_channels/pow(2, i+1)),
                                                  head_dims[i],
                                                  int(np.ceil(seq_lens[i]/seg_len_max)),
                                                  kernel_sizes[i+1]) for i in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = enc_inputs
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        return enc_outputs


class Decoder(nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(int(n_channels/pow(2, n_layers-i)),
                                                  int(n_channels/pow(2, n_layers-i-1)),
                                                  kernel_sizes[n_layers-i]) for i in range(n_layers)])

    def forward(self, dec_inputs):
        dec_outputs = dec_inputs
        for layer in self.layers:
            dec_outputs = layer(dec_outputs)
        return dec_outputs


class AE(nn.Module):
    """
    Autoencoder
    """

    def __init__(self, in_dim, z_dim=3, k_size=7, seg_max=500):
        super(AE, self).__init__()

        global seg_len_max
        seg_len_max = seg_max

        d = in_dim
        for i in range(n_layers+1):
            if d % 2 == 0:
                d = np.floor((d - k_size + 1) / 2 + 1)
                seq_lens.append(d)
                kernel_sizes.append(k_size-1)
            else:
                d = np.floor((d - k_size) / 2 + 1)
                seq_lens.append(d)
                kernel_sizes.append(k_size)
        d = np.int32(d)

        self.conv = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_sizes[0], stride=2),
            nn.LeakyReLU()
        )
        self.encoder = Encoder()
        self.fc1 = nn.Linear(int(d * n_channels/pow(2, n_layers)), z_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(z_dim, int(d * n_channels/pow(2, n_layers))),
            nn.LeakyReLU()
        )
        self.decoder = Decoder()
        self.ct1 = nn.ConvTranspose1d(n_channels, 1, kernel_sizes[0], stride=2)
        self.ct2 = nn.ConvTranspose1d(n_channels, 1, kernel_sizes[0], stride=2)

        initialize_weights(self)

    def encode(self, x):
        h = self.conv(x)
        h = self.encoder(h)
        h = h.view(h.size(0), -1)
        z = self.fc1(h)
        return z

    def decode(self, z):
        z = self.fc2(z)
        z = z.view(z.size(0), int(n_channels/pow(2, n_layers)), -1)
        z = self.decoder(z)
        mu = torch.exp(self.ct1(z).squeeze())
        sigma = torch.exp(self.ct2(z).squeeze())
        return mu, sigma

    def forward(self, x):
        z = self.encode(x)
        mu, sigma = self.decode(z)
        return z, mu, sigma
