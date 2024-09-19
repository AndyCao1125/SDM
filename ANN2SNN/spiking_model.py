import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from spiking_layer import Spiking, last_Spiking, IF, Spiking_TimeEmbed
from quant_layer import QuantSwish

class S_TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim, bit, spike_time: int = 3):
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

        self.timembedding = Spiking_TimeEmbed(nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            IF(),
            nn.Linear(dim, dim),
        ), spike_time, alpha_loc = 2)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()
        self.idem = False

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        if self.idem:
            return x
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.main(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()
        self.idem = False

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        if self.idem:
            return x
        B, T, C, H, W = x.shape
        x = x.flatten(0, 1)  ## [T*B C H W]
        # _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        _, C, H, W = x.shape
        x = x.reshape(B, T, C, H, W).contiguous()
        return x

class S_ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, bit, spike_time: int = 3):
        super().__init__()
        self.idem = False
        self.block1 = Spiking(nn.Sequential(
            nn.GroupNorm(32, in_ch),
            IF(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        ), spike_time, alpha_loc=1)
        self.temb_proj = Spiking(nn.Sequential(
            IF(),
            nn.Linear(tdim, out_ch),
        ),spike_time, alpha_loc=0)
        self.block2 = Spiking(nn.Sequential(
            nn.GroupNorm(32, out_ch),
            IF(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        ), spike_time, alpha_loc=1)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2.block[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        if self.idem:
            return x
        B, T, C, H, W = x.shape
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x.flatten(0,1)).reshape(B, T, -1, H, W).contiguous()
        return h


class S_UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, num_res_blocks, dropout, bit=32, spike_time: int = 3):
        super().__init__()
        tdim = ch * 4
        self.bit=bit
        self.spike_time = spike_time
        self.time_embedding = S_TimeEmbedding(T, ch, tdim, self.bit, self.spike_time)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(S_ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, bit=self.bit, spike_time=self.spike_time))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            S_ResBlock(now_ch, now_ch, tdim, dropout, self.bit, self.spike_time),
            S_ResBlock(now_ch, now_ch, tdim, dropout, self.bit, self.spike_time),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(S_ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, bit=self.bit, spike_time=self.spike_time))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = last_Spiking(nn.Sequential(
            nn.GroupNorm(32, now_ch),
            IF(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)), self.spike_time, alpha_loc=1)
        self.initialize()

        self.timembedding_idem = False
        self.head_idem = False
        self.tail_idem = False
        self.upblocks_target = False
    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail.block[-1].weight, gain=1e-5)
        init.zeros_(self.tail.block[-1].bias)

    def forward(self, x, t):
        if self.timembedding_idem:
            return t
        x_in = x
        t_in = t
        x_in = x_in.unsqueeze(1)
        x_in = x_in.repeat(1, self.spike_time, 1, 1, 1)
        # Timestep embedding
        temb = self.time_embedding(t_in)
        # Downsampling
        if self.head_idem:
            return temb
        train_shape = [x_in.shape[0], x_in.shape[1]]
        x_in = x_in.flatten(0, 1)
        h = self.head(x_in)
        train_shape.extend(h.shape[1:])
        h = h.reshape(train_shape)

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)

        # Upsampling
        for (idx,layer) in enumerate(self.upblocks):
            if layer.idem:
                if self.middleblocks[1].idem:
                    h = layer(h, temb)
                else:
                    if idx == 0 or (not self.upblocks[idx-1].idem and layer.idem):
                        if not self.upblocks_target:
                            if isinstance(layer, S_ResBlock):
                                h = torch.cat([h, hs.pop()], dim=2)
                            h = layer(h, temb)
                        else:
                            h = layer(h, temb)
                    else:
                        h = layer(h, temb)

            else:
                if isinstance(layer, S_ResBlock):
                    h = torch.cat([h, hs.pop()], dim=2)
                h = layer(h, temb)
        if self.tail_idem:
            return h
        h = self.tail(h)
        return h

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantSwish):
                m.show_params()


if __name__ == '__main__':
    import re
    import sys
    batch_size = 1
    model = S_UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2],num_res_blocks=2, dropout=0.1, bit=3)
    # load model and evaluate

    new_state_dict = {}
    old_state_dict = torch.load('./logs/RELU_QUANT2B_DDPM_CIFAR10_EPS/SNN_2.0bit_ft/ft_blocks_samples/model_best.pt')['ema_model']
    for key in old_state_dict:
        parts = key.split('.')
        if 'main' in key or 'shortcut' in key or 'head' in key:
            new_key = key
        else:
            parts.insert(-2, 'block')
            new_key = '.'.join(parts)
        new_state_dict[new_key] = old_state_dict[key]
    
    model.load_state_dict(old_state_dict, strict=True)
    print("Successfully load weight!")
    print(model.state_dict()['time_embedding.timembedding.block.2.act_alpha'])

    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)
    print(x.shape, t.shape)

    print(model.tail.idem)


    m=getattr(model,'downblocks')
    print(old_state_dict.keys())
    print(model.head_idem,model.time_embedding,model.tail_idem)
