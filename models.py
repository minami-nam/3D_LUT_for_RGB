import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Function
import tifffile as tiff
import numpy as np
import datasets as ds
import kornia.color as K
import trilinear_c__ext as trilinear_ext

# ---------------------- normalization2D 관련 ---------------------- 

def get_norm2d(num_features: int, norm_type: str = "instance"):

    if norm_type == "batch":
        return nn.BatchNorm2d(num_features)  # running stats ON by default
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")


# ---------------------- Debugging 및 Parameter ---------------------- 

torch.autograd.set_detect_anomaly(True)


# ----------------------  Weight 관련 ---------------------- 

def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# ---------------------- Classifier 및 Layer 관련 ---------------------- 

def discriminator_block(in_filters, out_filters, normalization=False, norm_type: str = "instance"):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
              nn.LeakyReLU(0.2)]
    if normalization:
        layers.append(get_norm2d(out_filters, norm_type))
    return layers


class Classifier(nn.Module):
    def __init__(self, in_channels=3, num_LUTS=3, norm_type: str = "instance"):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256,256), mode='bilinear'),
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            get_norm2d(16, norm_type),
            *discriminator_block(16,32, normalization=True, norm_type=norm_type),
            *discriminator_block(32,64, normalization=True, norm_type=norm_type),
            *discriminator_block(64,128, normalization=True, norm_type=norm_type),
            *discriminator_block(128,128, norm_type=norm_type),
            nn.Dropout(0.5),
            nn.Conv2d(128,num_LUTS,8)
        )

    def forward(self, img_input):
        return self.model(img_input)
    
# ---------------------- 3D LUT ---------------------- 

class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33, fname_3d="IdentityLUT33.txt", use_file=True):
        super(Generator3DLUT_identity, self).__init__()
        self.dim = dim
        if use_file:
            # 3D identity 파일을 읽어 베이스 3D LUT 생성
            with open(fname_3d, 'r') as f:
                lines = f.readlines()
            LUT3 = torch.zeros(3, dim, dim, dim)
            idx = 0
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        x = lines[idx].split()
                        LUT3[0, i, j, k] = float(x[0])
                        LUT3[1, i, j, k] = float(x[1])
                        LUT3[2, i, j, k] = float(x[2])
                        idx += 1
        else:
            # 파일 없이 수학적으로 identity 3D LUT 생성
            r = torch.linspace(0, 1, dim)
            g = torch.linspace(0, 1, dim)
            b = torch.linspace(0, 1, dim)
            R, G, B = torch.meshgrid(r, g, b, indexing='ij')
            LUT3 = torch.stack([R, G, B], dim=0)  # (3,dim,dim,dim)

        self.LUT = nn.Parameter(LUT3)


    def forward(self, x):
        return TrilinearInterpolation.apply(self.LUT, x)


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()
        LUT = torch.zeros(3,dim,dim,dim)
        self.LUT = nn.Parameter(LUT)

    def forward(self, x):
        return TrilinearInterpolation.apply(self.LUT, x)
    
# ---------------------- Trilinear Interpolation ---------------------- 

class TrilinearInterpolation(Function):
    @staticmethod
    def forward(ctx, LUT, x):  # 3D LUT
        x_cont   = x.contiguous()
        LUT_cont = LUT.contiguous()

        B, C, H, W = x_cont.shape
        dim     = LUT_cont.size(1)

        shift   = dim ** 3
        binsize = 1.0 / (dim - 1)

        # CUDA 바인딩 호출 (qforward)
        output = trilinear_ext.forward(
            LUT_cont, x_cont,
            dim, shift, binsize,
            W, H, B
        )

        # backward용 저장
        ctx.save_for_backward(x_cont, LUT_cont)
        ctx.dim      = dim
        ctx.shift    = shift
        ctx.binsize  = binsize
        ctx.W        = W
        ctx.H        = H
        ctx.B        = B

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, LUT = ctx.saved_tensors
        dim      = ctx.dim
        shift    = ctx.shift
        binsize  = ctx.binsize
        W, H, B= ctx.W, ctx.H, ctx.B

        grad_output = grad_output.contiguous()

        # backward는 LUT에 대한 그래디언트만 계산해 반환합니다: (3, D, D, D, DN)
        grad_LUT = trilinear_ext.backward(
            x, grad_output,
            dim, shift, binsize,
            W, H, B
        )

        # 입력 x에 대한 grad는 계산하지 않으면 None
        return grad_LUT, None


# ----------------------  Total Variation 관련 ---------------------- 

class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D, self).__init__()
        # Register weights as buffers so .to(device) moves them automatically
        w_r = torch.ones(3, dim, dim, dim-1)
        w_r[:,:,:,(0,dim-2)] *= 2.0
        self.register_buffer('weight_r', w_r)

        w_g = torch.ones(3, dim, dim-1, dim)
        w_g[:,:,(0,dim-2),:] *= 2.0
        self.register_buffer('weight_g', w_g)

        w_b = torch.ones(3, dim-1, dim, dim)
        w_b[:,(0,dim-2),:,:] *= 2.0
        self.register_buffer('weight_b', w_b)

        self.relu = nn.ReLU()

    def forward(self, lut_tensor):
        # Accept raw LUT tensor rather than module
        LUT = lut_tensor if not hasattr(lut_tensor, 'LUT') else lut_tensor.LUT
        #    weight_r.shape[0] == C (e.g. 3)
        if LUT.shape[0] != self.weight_r.shape[0]:
            # assume it’s [D, C, H, W] → move C→0
            LUT = LUT.permute(1, 0, 2, 3).contiguous()

        dif_r = LUT[:,:,:,:-1] - LUT[:,:,:,1:]
        dif_g = LUT[:,:,:-1,:] - LUT[:,:,1:,:]
        dif_b = LUT[:,:-1,:,:] - LUT[:,1:,:,:]
        tv = (torch.mean(dif_r**2 * self.weight_r) +
              torch.mean(dif_g**2 * self.weight_g) +
              torch.mean(dif_b**2 * self.weight_b))
        mn = (torch.mean(self.relu(dif_r)) +
              torch.mean(self.relu(dif_g)) +
              torch.mean(self.relu(dif_b)))
        return tv, mn
    
# ------------------- SSIM ------------------------------
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 입력 경계의 반사를 사용하여 상/하/좌/우에 입력 텐서를 추가로 채웁니다.
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x) 
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d), 0, 1)