import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from quant_layer import QuantReLU


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim, bit):
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
            QuantReLU(inplace=True, bit=bit),
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
        x = self.main(x)
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
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout,bit):
        super().__init__()
        self.idem = False
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            QuantReLU(inplace=True, bit=bit),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            QuantReLU(inplace=True, bit=bit),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            QuantReLU(inplace=True, bit=bit),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
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
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        if self.idem:
            return x
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, num_res_blocks, dropout,bit=32):
        super().__init__()
        tdim = ch * 4
        self.bit=bit
        self.time_embedding = TimeEmbedding(T, ch, tdim, self.bit)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, bit=self.bit))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, self.bit),
            ResBlock(now_ch, now_ch, tdim, dropout, self.bit),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, bit=self.bit))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            QuantReLU(inplace=True, bit=self.bit),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

        self.timembedding_idem = False
        self.head_idem = False
        self.tail_idem = False
        self.upblocks_target = False
    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        if self.timembedding_idem:
            return t
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        if self.head_idem:
            return temb
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for (idx, layer) in enumerate(self.upblocks):
            if layer.idem:
                if self.middleblocks[1].idem:
                    h = layer(h, temb)
                else:
                    if idx == 0 or (not self.upblocks[idx-1].idem and layer.idem):
                        if not self.upblocks_target:
                            if isinstance(layer, ResBlock):
                                h = torch.cat([h, hs.pop()], dim=1)
                            h = layer(h, temb)
                        else:
                            h = layer(h, temb)
                    else:
                        h = layer(h, temb)
            else:
                if isinstance(layer, ResBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb)
        if self.tail_idem:
            return h
        h = self.tail(h)

        return h

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantReLU):
                m.show_params()


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2],num_res_blocks=2, dropout=0.1,bit=32)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y)