import math
import copy
from pathlib import Path
from random import random
from functools import partial   # Partial functions allow us to fix a certain number of arguments of a function and generate a new function
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce    # einops is an good package for us to do deep-learning project with diffferent frackorks, like pytorch and tensor-flow
from einops.layers.torch import Rearrange  # worked for pytorch layers
from tqdm.auto import tqdm
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):  # in the up-samping process for the unet.
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'), # double the input size
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1) # Hin = Hout, but will double the channel of the output
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),  # this is the downsamplng network by max-plooing method with kernel_size = (2, 2)
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)  # Hin = Hout, just change the output channels
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):   # similar regulation method for weightstandardizedConv2d()
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):  # this will work to first normalize the output before feed into other networks
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds  # （正铉曲线位置插入）here is for the time-sequential embedding

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

class RandomOrLearnedSinusoidalPosEmb(nn.Module): # zheng xuan quan xian （正铉曲线）
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False): #dim =16,
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random) # This just create one container that does not change the input tensor.

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))   # noise_level[:, None] == noise_level.unsqueeze(1)
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)  # Hin = Hout
        self.norm = nn.GroupNorm(groups, dim_out)  # use groups normalization rather than Batch normalization
        self.act = nn.SiLU()  # sigmoid linear Unit(SiLU) functions. element wiee

    def forward(self, x, scale_shift = None):  #block with time-bedding
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

'''
# old
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),  # sigmoid Linear Unit(SiLU) functions, element wise
            nn.Linear(time_emb_dim, dim_out * 2)   # time_emb_dim = dim * 4 = 256
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)  # weighted convolution block
        self.block2 = Block(dim_out, dim_out, groups = groups) # weighted convolution block
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None  # the feature of time_emb = dim * 4
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            print(f'------the time embeding shape is {time_emb.shape}-----------')
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
'''

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0, use_affine_level=False, groups=8):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            time_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module): # Hin= Hout with regular Conv2d but has layerNorm for the Linear-multihead attention in the down/up sampling process in the unet
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)  # the convolution with  kernel_size = 1 only can lead to the channel's change

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):  # Hin = Hout for a regular multihead attention with layerNorm in central block of the unet
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):  # this is residual-multihead attention Unet which is similar to previous R2Attu_unet
    def __init__(
        self,
        dim,   # here dim = 64
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,  # this is image, so for me the channels is 1 for me
        self_condition = True,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        noise_level_emb = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)  # Hout = Hin

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) # in_out is also determined by the dim, so the time_dim is simiar to time_dim
        # in_out = [(dim, dim), (dim, 2*dim), (2*dim, 4*dim), (4*dim, 8*dim)]
        # so if dim = 64, in_out = [(64, 64), (64, 128), (128, 256), (256, 512)]

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)  # this is the residual-block with time-embedding parameters

        # time embeddings
        time_dim = dim * 4  # time_dim is determined by the dim, so if dim is not none

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.noise_level_emb = noise_level_emb

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features) # first argument=16, second = boolean
            fourier_dim = learned_sinusoidal_dim + 1
        elif self.noise_level_emb:
            #print("----------come here please pay attention -----------\n")
            sinu_pos_emb = PositionalEncoding(dim)
            fourier_dim = dim
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),  # this is the Gaussian Error Linear Units functions
            nn.Linear(time_dim, time_dim)   # here make the t = dim * 4
        )

        # layers

        self.downs = nn.ModuleList([])  # down-process
        self.ups = nn.ModuleList([])   # up-rocess
        num_resolutions = len(in_out) # here the len(in_out) = 4;

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),  # Hin = Hout the block in fact is also the residual block
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),  # Hin = Hout
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),  # Attehntion layer that has normalization layer before multihead attention with residual thought the whole as the normalization block
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)  # down-sampling
            ]))

        # below is the middle/centra block is the multihead_aatention
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))  # The pre_normalized multihead attention of residual
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)   # final-residual-block
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1) #  normal Conv2d to change the output channels

    def forward(self, x, time, x_self_cond = None):
        # if self.self_condition:
            # x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        x = torch.cat((x_self_cond, x), dim = 1) if self.self_condition else x

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)  # here is the time-embedding 4*dim
        # print(f'\nthe time in forward original time is {time.shape} and after embedding is {t.shape}----\n')

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)  # so the time embedding always is embedded in the Residual_block or blocks
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class
# extract() to get the alpha_t at time t, then get the xt based on x0 and alpha_t
def extract(a, t, x_shape):  # t has same dimension as a, in order to get the corresponding beta_t or alhpa_t to caculate the xt at time t
    b, *_ = t.shape   # a has the length as numsteps
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # this will make the a has the shape (b, 1, 1, 1)

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l2',
        objective = 'pred_noise',
        beta_schedule = 'linear',
        schedule_fn_kwargs = dict(),
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.,
        auto_normalize = False
    ):  # This part is to initialize some parameters for model training
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim) # make sure channels == out_dim
        assert not model.random_or_learned_sinusoidal_cond  # model.random_or_learned_sinusoidal_cond = False

        self.model = model
        self.channels = self.model.channels # here is 1 for the model parameters
        self.self_condition = self.model.self_condition  # in model is none

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)  # in betas in the DDPM model

        alphas = 1. - betas  # alphas in the DDPM model
        alphas_cumprod = torch.cumprod(alphas, dim=0)   # the cumulative product of elements of input in the dimension dim
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)  # pad the alphas_cumprod in the beginning with value = 1.0
        self.sqrt_alphas_cumprod_prev = torch.sqrt(F.pad(alphas_cumprod_prev, (1, 0), value=1.))

        timesteps, = betas.shape  # betas in fact is a 1-D array
        self.num_timesteps = int(timesteps)   # 1000
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps   # self.is_ddim_sampling = false
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32  # below is first to initialize the built-in register_buffer function in nn.model modules

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32)) # note this parameters wrapped by regise=ter_function will not change or be trained by the model

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))  # the base for torch.log is natural number e
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # this is 后验方差，betas at time t in equal(7)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))  # coef1 before x0 in equal (7)
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))  # coef2 before xt in euqual (7)

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise


    def predict_noise_from_start(self, x_t, t, x0):
        return (self.sqrt_recip_alphas_cumprod[t] * x_t - x0) / \
            self.sqrt_recipm1_alphas_cumprod[t]


    def predict_v(self, x_start, t, noise):
        return self.sqrt_alphas_cumprod[t] * noise - self.sqrt_one_minus_alphas_cumprod[t] * x_start

    def predict_start_from_v(self, x_t, t, v):
        return self.sqrt_alphas_cumprod[t] * x_t - self.sqrt_one_minus_alphas_cumprod[t] * v

    def q_posterior(self, x_start, x_t, t): # same  # this is somethng that for equal 6 which will be used in p_mean_variance() function
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t

        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, t_real = None):
        model_output = self.model(x, time = t, x_self_cond = x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity  # this is to give return the torch,clamp functions with fixed parameters

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t_real, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t_real, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t_real, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t_real, x_start)

        return ModelPrediction(pred_noise, x_start)  # note, the ModelPrediction() is a container which could contains the result by keys

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True): # predict x, mean and variance
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        preds = self.model_predictions(x, t = noise_level, x_self_cond = x_self_cond, t_real = t)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start   # the last one item x_start is extra information

    @torch.no_grad()  # below will introduce the p_mean_variance() function which contains the q_posterior() function
    def p_sample(self, x, t, x_self_cond = None):
        # b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x) # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise  # torch.exp(tensor) == tensor.exp()
        return pred_img, x_start

    @torch.no_grad()  # p_sample() is used in the below function
    def p_sample_loop(self, x_in, return_all_timesteps = False):
        device = self.betas.device
        if self.self_condition:
            x_start = x_in  # x_in is the noisy data in dataloader so x_start should also be modified
            img = torch.randn(x_start.shape, device=device)
            self_cond = x_start
            ret = [x_start]
        else:
            shape = x_in
            img = torch.randn(shape, device=device)
            self_cond = None
            ret = [img]
        print(f'\nthe image shape is {img.shape} and ret shape is {len(ret)}')
        # imgs = [img]
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, _ = self.p_sample(img, t, self_cond)
            ret.append(img)
        print(f'\nafter loop the image shape is {img.shape} and ret shape is {len(ret)}')

        ret = ret[-1] if not return_all_timesteps else ret

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()  # here has not been modified, but should be modified
    def ddim_sample(self, shape, return_all_timesteps = False):  # this will use the Model to predict the result
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # sampling_timesteps = none, so steps = none + 1 == 1; total_timesteps = T==1000
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None   # self_cond is always None, only for the first iteration
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()  # for equal-8 in algorithm 2 in DDPM article
            c = (1 - alpha_next - sigma ** 2).sqrt()   # for equal-8 in algorithm 2 in DDPM article

            noise = torch.randn_like(img)

            # below is about equations in the algorithm 2
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()   # below valuate() function will introduce ddim_sample() and p_sample_loop() this two function
    def sample(self, x, return_all_timesteps = False):   # when using sample. you should set conditinal=False, this is the unconditional situation
        batch_size = x.shape[0]
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample   # not self.is_ddim_sampling is true if sampling_timesteps = none in __init__()
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    @torch.no_grad()  # for super_resolution image
    def super_resolution(self, x_in, continous=False):   # here x_in is nosiy data in the dataloader
        return self.p_sample_loop(x_in, continous)

    # below is the forward process to valuate the current from noise, function is used in p_losses()
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):  # here noise is gaussion  noise which is used for forward-adding noising valuate
        noise = default(noise, lambda: torch.randn_like(x_start)) # noise has the same shape as x_start

        return (continuous_sqrt_alpha_cumprod * x_start  +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_in, t = None, noise = None):   # here the x_in the whole dataloader in datasets
        x_start, x_end = x_in
        b, c, h, w = x_end.shape
        assert h == self.image_size and w == self.image_size, f'height and width of image must be {self.image_size}'
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_end.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_end))  # here is the gaussion noise

        # noise valuate
        x = self.q_sample(x_start = x_end, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise = noise)  # here get x from gaussion noise sampling

        # condition value
        x_self_cond = x_start if self.self_condition else None

        # predict and take gradient step
        model_out = self.model(x, time = continuous_sqrt_alpha_cumprod, x_self_cond = x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        # print(f'the direct output loss shape is {loss.shape}')

        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # print(f'the loss shape is {loss.shape} and the p2_loss_weight is {self.p2_loss_weight.shape}')
        # t1 = torch.tensor([t]).long().repeat(b).to(x_end.device)
        # loss = loss * extract(self.p2_loss_weight, t1, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        img = self.normalize(img)
        return self.p_losses(img, *args, **kwargs)
