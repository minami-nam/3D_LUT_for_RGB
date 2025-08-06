import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Function, Variable
import numpy as np
import math
import datasets as ds
# C++/CUDA 확장 모듈을 trilinear_ext로 임포트
import trilinear_c._ext as trilinear_ext

import os

torch.autograd.set_detect_anomaly(True)

def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class resnet18_224(nn.Module):
    def __init__(self, out_dim=5, aug_test=False):
        super(resnet18_224, self).__init__()
        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)
        net.fc = nn.Linear(512, out_dim)
        self.upsample = nn.Upsample(size=(224,224), mode='bilinear')
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            x = torch.cat((x, torch.flip(x, [3])), 0)
        return self.model(x)


##############################
#           DPE
##############################

class RGBNIRtoRGB(nn.Module):
    def __init__(self):
        super().__init__()
        # Kernel_size를 1로 하면, 3x3 배열에서 손실 없이 3x3 배열로 복제가 가능하고, 대신 채널만 절반으로 줄어드는 효과를 볼 수 있음.
        # 일단 선형 연산인 conv를 시도 후, 이를 비선형 활성화(ReLU)를 통해 복잡한 함수를 학습할 수 있게 도움.
        
        self.conv = nn.Conv2d(4, 8, kernel_size=1, padding=0)
        self.final = nn.Conv2d(8, 3, kernel_size=1)
        
    def forward(self, RGB, NIR):
        x = torch.cat([RGB, NIR], dim=1)
        x = F.relu(self.conv(x))
        return self.final(x)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 5, 2, 2),
                  nn.SELU(inplace=True)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.SELU(inplace=True),
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,16,in_channels,padding=1),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(16, affine=True),
        )
        self.down1 = UNetDown(16,32)
        self.down2 = UNetDown(32,64)
        self.down3 = UNetDown(64,128)
        self.down4 = UNetDown(128,128)
        self.down5 = UNetDown(128,128)
        self.down6 = UNetDown(128,128)
        self.down7 = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(128,128,1)
        )
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv1x1 = nn.Conv2d(256,128,1)
        self.up1 = UNetUp(128,128)
        self.up2 = UNetUp(256,128)
        self.up3 = UNetUp(192,64)
        self.up4 = UNetUp(96,32)
        self.final = nn.Sequential(
            nn.Conv2d(48,16,3,padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(16,out_channels,3,padding=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.upsample(d7)
        d9 = self.conv1x1(torch.cat((d4,d8),1))
        u1 = self.up1(d9, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, x1)
        return torch.add(self.final(u4), x)


class Discriminator_UNet(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator_UNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16,32),
            *discriminator_block(32,64),
            *discriminator_block(64,128),
            *discriminator_block(128,128),
            *discriminator_block(128,128),
            nn.Conv2d(128,1,4)
        )

    def forward(self, img_input):
        return self.model(img_input)


def discriminator_block(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters,out_filters,3,stride=2,padding=1),
              nn.LeakyReLU(0.2)]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256,256), mode='bilinear'),
            nn.Conv2d(3,16,3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16,32),
            *discriminator_block(32,64),
            *discriminator_block(64,128),
            *discriminator_block(128,128),
            nn.Conv2d(128,1,8)
        )

    def forward(self, img_input):
        return self.model(img_input)


class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256,256), mode='bilinear'),
            nn.Conv2d(3,16,3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16,32, normalization=True),
            *discriminator_block(32,64, normalization=True),
            *discriminator_block(64,128, normalization=True),
            *discriminator_block(128,128),
            nn.Dropout(0.5),
            nn.Conv2d(128,3,8)
        )

    def forward(self, img_input):
        return self.model(img_input)


class Classifier_unpaired(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier_unpaired, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256,256), mode='bilinear'),
            nn.Conv2d(in_channels,16,3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16,32),
            *discriminator_block(32,64),
            *discriminator_block(64,128),
            *discriminator_block(128,128),
            nn.Conv2d(128,3,8)
        )

    def forward(self, img_input):
        return self.model(img_input)


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        fname = "IdentityLUT33.txt" if dim==33 else "IdentityLUT64.txt"
        with open(fname,'r') as file:
            LUT_lines = file.readlines()
        LUT = torch.zeros(3,dim,dim,dim)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    n = i*dim*dim + j*dim + k
                    x = LUT_lines[n].split()
                    LUT[0,i,j,k] = float(x[0])
                    LUT[1,i,j,k] = float(x[1])
                    LUT[2,i,j,k] = float(x[2])
        self.LUT = nn.Parameter(LUT)

    def forward(self, x):
        return TrilinearInterpolation.apply(self.LUT, x)


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()
        LUT = torch.zeros(3,dim,dim,dim)
        self.LUT = nn.Parameter(LUT)

    def forward(self, x):
        return TrilinearInterpolation.apply(self.LUT, x)


class TrilinearInterpolation(Function):
    @staticmethod
    def forward(ctx, LUT, x):
        # contiguity 보장
        x_cont = x.contiguous()
        LUT_cont = LUT.contiguous()
        
        # 3) contiguity 여부 출력 / assert
        print(f"[DEBUG] x_cont.is_contiguous={x_cont.is_contiguous()}, strides={x_cont.stride()}")
        print(f"[DEBUG] LUT_cont.is_contiguous={LUT_cont.is_contiguous()}, strides={LUT_cont.stride()}")
        assert x_cont.is_contiguous(),   "x_cont 가 contiguous 가 아닙니다!"
        assert LUT_cont.is_contiguous(), "LUT_cont 가 contiguous 가 아닙니다!"
        
        output = x_cont.new_zeros(x_cont.size())
        
        # 3) 연속성(assert) 점검
        assert x_cont.is_contiguous(), \
            f"x_cont not contiguous! strides={x_cont.stride()}"
        assert LUT_cont.is_contiguous(), \
            f"LUT_cont not contiguous! strides={LUT_cont.stride()}"


        # 파라미터
        B, C, H, W = x_cont.shape
        dim     = LUT.size(-1)
        shift   = dim ** 3
        binsize = 1.0001 / (dim - 1)

        torch.cuda.synchronize()

        # CUDA/CPU 모두 같은 바인딩 함수명을 쓰되, 
        # dispatch는 텐서가 GPU 상에 있느냐로 구분합니다.
        trilinear_ext.forward(
            LUT_cont, x_cont, output,
            dim, shift, binsize, W, H, B
        )


        # backward 에 필요하니 저장
        ctx.save_for_backward(x_cont, LUT_cont)
        ctx.dim     = dim
        ctx.shift   = shift
        ctx.binsize = binsize
        ctx.W       = W
        ctx.H       = H
        ctx.B       = B


        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, LUT = ctx.saved_tensors
        dim, shift, binsize = ctx.dim, ctx.shift, ctx.binsize
        W, H, B             = ctx.W, ctx.H, ctx.B
  

        # LUT gradient 만 계산
        grad_LUT = torch.zeros_like(LUT)


        trilinear_ext.backward(
            x, grad_output, grad_LUT,
            dim, shift, binsize, W, H, B
        )


        # (grad_wrt_LUT, grad_wrt_x) — x 에 대한 grad 는 None
        return grad_LUT, None


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
