import math
import torch
from abc import abstractmethod
import numpy as np
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from spikingjelly.activation_based import functional, surrogate
# from spikingjelly.activation_based.neuron import LIFNode as IFNode
# from spikingjelly.activation_based.neuron import ParametricLIFNode as IFNode
from spikingjelly.activation_based.neuron import IFNode as IFNode
from collections import OrderedDict
import os

global_thres = 1

class TSMConv2d(nn.Module):
    def __init__(self, timestep, in_ch, out_ch, kernel_size=3, stride=1,padding=1):
        super(TSMConv2d, self).__init__()
        self.T = timestep
        self.tsmconv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.tsmp = nn.Parameter(torch.ones(self.T, 1, 1, 1, 1))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.tsmconv.weight)
        init.zeros_(self.tsmconv.bias)

    def forward(self, input):
        # _, C, H, W = input.shape
        y = self.tsmconv(input)   ## [T*B C H W]
        _, C, H, W = y.shape
        y = y.reshape(self.T, -1, C, H, W).contiguous()
        y = y * self.tsmp
        y = y.flatten(0,1)
        # Multiply by the learnable parameter p
        return y



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):  ## T: total step of diff; d_model: base channel num; dim:d_model*4
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class Spk_DownSample(nn.Module):
    def __init__(self, in_ch, timestep=4, global_thres=1.0, use_cupy=False):
        super().__init__()
        self.conv = TSMConv2d(timestep, in_ch, in_ch, 3, stride=2, padding=1)
        # self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.neuron = IFNode(v_threshold=global_thres, surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        # self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape
        x = self.neuron(x)
        x = x.flatten(0, 1)  ## [T*B C H W]
        x = self.conv(x)
        _, C, H, W = x.shape
        x = self.bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class Spk_UpSample(nn.Module):
    def __init__(self, in_ch, timestep=4, global_thres=1.0, use_cupy=False):
        super().__init__()
        self.conv = TSMConv2d(timestep, in_ch, in_ch, 3, stride=1, padding=1)
        # self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.neuron = IFNode(v_threshold=global_thres, surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        # self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape
        x = self.neuron(x)
        x = x.flatten(0, 1)  ## [T*B C H W]
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        _, C, H, W = x.shape
        x = self.bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class Spk_ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, timestep=4, global_thres=1.0, use_cupy=False):
        super().__init__()
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.neuron1 = IFNode(v_threshold=global_thres,surrogate_function=surrogate.ATan())
        self.conv1 = TSMConv2d(timestep, in_ch, out_ch, 3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.neuron2 = IFNode(v_threshold=global_thres,surrogate_function=surrogate.ATan())
        self.conv2 = TSMConv2d(timestep, out_ch, out_ch, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)


        self.in_ch = in_ch
        self.out_ch = out_ch

        if in_ch != out_ch:
            self.shortcut = TSMConv2d(timestep, in_ch, out_ch, 1, stride=1, padding=0)
            # self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = Spike_SelfAttention(out_ch)
        else:
            self.attn = nn.Identity()


        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        # init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape

        h = self.neuron1(x)
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.conv1(h)
        h = self.bn1(h).reshape(T, B, -1, H, W).contiguous()

        temp = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, h.shape[-2], h.shape[-1])
        h = torch.add(h, temp)

        h = self.neuron2(h)
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.conv2(h)
        h = self.bn2(h).reshape(T, B, -1, H, W).contiguous()


        h = h + self.shortcut(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()

        h = self.attn(h)

        return h


class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    """

    def __init__(self, timestep=4) -> None:
        super().__init__()
        self.n_steps = timestep

    def forward(self, x):
        """
        x : (T,N,C,H,W)
        """

        arr = torch.arange(self.n_steps - 1, -1, -1)
        coef = torch.pow(0.8, arr).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x.device)
        out = torch.sum(x * coef, dim=0)
        return out


class Spk_UNet_TSM(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, timestep, img_ch=3, global_thres=1.0, use_cupy=False):
        super().__init__()
        # assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)  ## T: total step of diff; ch: base channel num; tdim:ch*4
        self.timestep = timestep  ## SNN timestep
        self.conv = TSMConv2d(self.timestep, img_ch, ch, kernel_size=3, stride=1, padding=1)
        # self.conv = nn.Conv2d(img_ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(ch)

        self.neuron = IFNode(v_threshold=global_thres,surrogate_function=surrogate.ATan())
        self.conv_identity = TSMConv2d(timestep, ch, ch, kernel_size=3, stride=1, padding=1)
        # self.conv_identity = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn_identity = nn.BatchNorm2d(ch)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(Spk_ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), timestep=self.timestep, global_thres=global_thres, use_cupy=use_cupy))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(Spk_DownSample(now_ch, self.timestep, global_thres=global_thres, use_cupy=use_cupy))
                chs.append(now_ch)
        # print(f'structure:{chs}')
        self.middleblocks = nn.ModuleList([
            Spk_ResBlock(now_ch, now_ch, tdim, dropout, attn=False, timestep=self.timestep, global_thres=global_thres, use_cupy=use_cupy),
            Spk_ResBlock(now_ch, now_ch, tdim, dropout, attn=False, timestep=self.timestep, global_thres=global_thres, use_cupy=use_cupy),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(Spk_ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), timestep=self.timestep, global_thres=global_thres, use_cupy=use_cupy))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(Spk_UpSample(now_ch, self.timestep, global_thres=global_thres, use_cupy=use_cupy))
        assert len(chs) == 0

        self.tail_bn = nn.BatchNorm2d(now_ch)
        self.tail_swish = Swish()
        self.tail_conv = TSMConv2d(self.timestep, now_ch, img_ch, kernel_size=3, stride=1, padding=1)


        self.T_output_layer = nn.Conv3d(img_ch, img_ch, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.last_bn = nn.BatchNorm2d(img_ch)
        self.swish = Swish()
        self.membrane_output_layer = MembraneOutputLayer(timestep=self.timestep)


        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x, t):

        x = x.unsqueeze(0).repeat(self.timestep, 1, 1, 1, 1)  # [T, B, C, H, W]

        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        T, B, C, H, W = x.shape
        h = x.flatten(0, 1)  ## [T*B C H W]
        h = self.conv(h)
        h = self.bn(h).reshape(T, B, -1, H, W).contiguous()
        h = self.neuron(h)

        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.conv_identity(h)
        h = self.bn_identity(h).reshape(T, B, -1, H, W).contiguous()

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, Spk_ResBlock):
                h = torch.cat([h, hs.pop()], dim=2)
            h = layer(h, temb)

        T, B, C, H, W = h.shape
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.tail_bn(h)
        h = self.tail_swish(h)
        h = self.tail_conv(h).reshape(T, B, -1, H, W).contiguous()

        h_temp = h.permute(1, 2, 3, 4, 0)  # [ B, C, H, W, T]
        h_temp = self.T_output_layer(h_temp).permute(4, 0, 1, 2, 3)   # [ T, B, C, H, W]
        h_temp = self.last_bn(h_temp.flatten(0,1)).reshape(T, B, -1, H, W).contiguous()
        h = self.swish(h_temp) + h  # [ T, B, C, H, W]

        h = self.membrane_output_layer(h)


        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 1
    model = Spk_UNet_TSM(
        T=1000, ch=128, ch_mult=[1, 2, 2, 4], attn=[8],
        num_res_blocks=2, dropout=0.1, timestep=8, global_thres=1., use_cupy=False).cuda()

