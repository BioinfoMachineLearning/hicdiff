import torch
import torch.nn as nn

n_feat = 256
kernel_size = 3



# original hicedrn
class _Res_Block(nn.Module):
    def __init__(self):
        super(_Res_Block, self).__init__()

        self.res_conv = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        y = self.relu(self.res_conv(x))
        y = self.res_conv(y)
        y *= 0.1
        y = torch.add(y, x)
        return y


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        in_ch = 1
        number_blocks = 32
        self.head = nn.Conv2d(in_ch, n_feat, kernel_size, padding = 1)

        self.body = self.make_layer(_Res_Block, number_blocks)

        self.tail = nn.Conv2d(n_feat, in_ch, kernel_size, padding=1)

    def make_layer(self, block, layers):
        res_block = []
        for _ in range(layers):
            res_block.append(block())
        res_block.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding = 1))

        return nn.Sequential(*res_block)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        out = self.tail(res)

        return out

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)


'''modified hicedrn_Diff that combined the time embedding into our previous model
 to make the model dynamics
'''

from einops import rearrange
import math
from functools import partial

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

class LayerNorm(nn.Module):
    def __init__(self, dim, scale = True, normalize_dim = 2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1

        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        normalize_dim = self.normalize_dim
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = normalize_dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = normalize_dim, keepdim = True)
        return (x - mean) * var.clamp(min = eps).rsqrt() * scale

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = LayerNorm(dim, normalize_dim = 1)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim, normalize_dim = 1)
        )

    def forward(self, x):
        residual = x

        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out) + residual

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim = n_feat, dim_out = n_feat, *, time_emb_dim = n_feat * 4): # dim == dim_out
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.conv = Block(dim, dim_out)
        self.act =  nn.SiLU()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.conv(x, scale_shift = scale_shift)
        h = self.act(h)
        h = self.conv(h)
        h *= 0.1
        h = torch.add(h, self.res_conv(x))

        return h

class hicedrn_Diff(nn.Module):
    def __init__(self,
                 channels = 1,
                 out_dim = None,
                 number_resnet = 32,
                 self_condition = False,
                 learned_sinusoidal_cond=False,
                 learned_sinusoidal_dim=16
                 ):
        super().__init__()
        self.channels = channels
        in_ch = channels * (2 if self_condition else 1)
        number_blocks = number_resnet
        self.self_condition = self_condition

        self.head = nn.Conv2d(in_ch, n_feat, kernel_size, padding = 1)

        # time embeddings
        time_dim = n_feat * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)  # first argument=16, second = boolean
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(n_feat)
            fourier_dim = n_feat

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),  # this is the Gaussian Error Linear Units functions
            nn.Linear(time_dim, time_dim)  # here make the t = n_feat * 4
        )

        block_res = partial(ResnetBlock, dim = n_feat, dim_out = n_feat, time_emb_dim = time_dim)
        self.body = self.make_layer(block_res, number_blocks)
        #self.body = nn.ModuleList([])

        '''
        for i in range(number_blocks):
            self.body.append(nn.ModuleList([ResnetBlock(n_feat, n_feat, time_emb_dim = time_dim)]))
        '''
        self.body_tail = nn.Conv2d(n_feat, n_feat, kernel_size, padding = 1)

        default_out_channels = channels
        self.out_dim = default(out_dim, default_out_channels)
        self.tail = nn.Conv2d(n_feat, self.out_dim, kernel_size, padding=1)

    def make_layer(self, block, layers):
        res_block = []
        for i in range(layers):
            res_block.append(block())
        #res_block.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding = 1))

        return nn.Sequential(*res_block)

    def forward(self, x, time, x_self_cond = None):
        '''
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        '''
        x = torch.cat((x_self_cond, x), dim=1) if self.self_condition else x

        x = self.head(x)
        r = x.clone()
        t = self.time_mlp(time)  # here is the time-embedding n_feat * 4

        #res = self.body(x, t)
        for block in self.body:
            x = block(x, t)
        x = self.body_tail(x)
        x += r
        #res = self.body_tail(res)
        #res += r

        out = self.tail(x)

        return out

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)