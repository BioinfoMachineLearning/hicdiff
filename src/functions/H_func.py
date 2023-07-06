import torch
import numpy as np

def MakeFunc(deg = 'deno', image_channel = 1, image_size = 64, device = None):
    H_funcs = None
    if deg[:2] == 'cs':
        compress_by = int(deg[2:])
        from src.functions.svd_replacement import WalshHadamardCS
        H_funcs = WalshHadamardCS(image_channel, image_size, compress_by, torch.randperm(image_size ** 2, device=device), device)
    elif deg[:3] == 'inp':
        from src.functions.svd_replacement import Inpainting
        if deg == 'inp_mask':
            missing_r = torch.randperm(image_size ** 2)[:image_size ** 2 // 2].to(device).long()
        #missing_g = missing_r + 1
        #missing_b = missing_g + 1
        #missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        H_funcs = Inpainting(image_channel, image_size, missing_r, device)
    elif deg == 'deno':  # denoise
        from src.functions.svd_replacement import Denoising
        H_funcs = Denoising(image_channel, image_size, device)
    elif deg[:10] == 'sr_bicubic':
        factor = int(deg[10:])
        from src.functions.svd_replacement import SRConv
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        H_funcs = SRConv(kernel / kernel.sum(), image_channel, image_size, device, stride=factor)
    elif deg == 'deblur_uni':   # this is the one tested in super-image restoration.
        from src.functions.svd_replacement import Deblurring
        H_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(device), image_channel, image_size, device)
    elif deg == 'deblur_gauss':
        from src.functions.svd_replacement import Deblurring
        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        H_funcs = Deblurring(kernel / kernel.sum(), image_channel, image_size, device)
    elif deg == 'deblur_aniso':  # Anisotropic Deblurring  各项异性去模糊
        from src.functions.svd_replacement import Deblurring2D
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), image_channel, image_size, device)
    elif deg[:2] == 'sr':  # super-resolution has three factor 2, 4, 8;
        blur_by = int(deg[2:])
        from src.functions.svd_replacement import SuperResolution
        H_funcs = SuperResolution(image_channel, image_size, blur_by, device)
    elif deg == 'color':
        from src.functions.svd_replacement import Colorization
        H_funcs = Colorization(image_size, device)
    else:
        print("ERROR: degradation type not supported")
        quit()

    return H_funcs