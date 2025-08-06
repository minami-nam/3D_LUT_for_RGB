import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from pytorch_msssim import ssim

from multiprocessing import freeze_support

import torchvision.transforms as transforms
from torch.utils.data import DataLoader




from models import *
from datasets import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--dataset_name", type=str, default="fiveK")
    parser.add_argument("--input_color_space", type=str, default="sRGB")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--lambda_smooth", type=float, default=1e-4)
    parser.add_argument("--lambda_monotonicity", type=float, default=10.0)
    parser.add_argument("--n_cpu", type=int, default=6)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT")
    opt = parser.parse_args()
    opt.output_dir += '_' + opt.input_color_space
    os.makedirs(f"saved_models/{opt.output_dir}", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    criterion = nn.MSELoss().to(device)
    # Initialize models
    LUT0 = Generator3DLUT_identity().to(device)
    LUT1 = Generator3DLUT_zero().to(device)
    LUT2 = Generator3DLUT_zero().to(device)
    classifier = Classifier().to(device)
    TV3 = TV_3D().to(device)
    TV3.weight_r = TV3.weight_r.to(device)
    TV3.weight_g = TV3.weight_g.to(device)
    TV3.weight_b = TV3.weight_b.to(device)
    trilinear_ = TrilinearInterpolation()

    # Optimizer
    optimizer_G = torch.optim.Adam(
        itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()),
        lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    start_epoch = 0
        
    # Load from checkpoint if needed
    if opt.epoch > 0:
        # load model
        LUTs = torch.load(f"saved_models/{opt.output_dir}/LUTs_{opt.epoch}.pth", map_location=device)
        LUT0.load_state_dict(LUTs['0'])
        LUT1.load_state_dict(LUTs['1'])
        LUT2.load_state_dict(LUTs['2'])
        classifier.load_state_dict(torch.load(f"saved_models/{opt.output_dir}/classifier_{opt.epoch}.pth", map_location=device))
        # load optimizer state
        opt_sd = torch.load(f"saved_models/{opt.output_dir}/optimizer_{opt.epoch}.pth", map_location=device)
        optimizer_G.load_state_dict(opt_sd)
        # relocate optimizer state tensors
        for state in optimizer_G.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = opt.epoch
    else:
        classifier.apply(weights_init_normal_classifier)
        torch.nn.init.constant_(classifier.model[16].bias, 1.0)

    # Data loaders
    Dataset = ImageDataset_sRGB if opt.input_color_space=='sRGB' else ImageDataset_XYZ
    train_loader = DataLoader(Dataset(f"data/{opt.dataset_name}", mode="train"),
                              batch_size=1, shuffle=True, num_workers=4)
    test_loader  = DataLoader(Dataset(f"data/{opt.dataset_name}", mode="test"),
                              batch_size=1, shuffle=False, num_workers=4)

    def generator_train(img): 
        with torch.no_grad():
            pred = classifier(img)
        gen_A0 = LUT0(img); gen_A1 = LUT1(img); gen_A2 = LUT2(img)
        wnorm = pred.pow(2).mean()
        w = pred.view(pred.size(0),3,1,1)
        stacked = torch.stack([gen_A0,gen_A1,gen_A2],dim=1)
        combine = (w.unsqueeze(2) * stacked).sum(1)
        return combine, wnorm

    def generator_eval(img):
        with torch.no_grad():
            # 예측된 weight: pred.shape == [B, 3] 또는 [3]
            pred = classifier(img).squeeze()

            # 배치 없는 경우 → 배치 1로 reshape
            if pred.ndim == 1 and pred.shape[0] == 3:
                pred = pred.unsqueeze(0)   # shape → [1, 3]

            B = pred.shape[0]  # 배치 크기

            # LUTs: [3, 33, 33, 33] × 3 → [3, 3, 33, 33, 33]
            LUTs = torch.stack([
                LUT0.LUT,
                LUT1.LUT,
                LUT2.LUT
            ], dim=0).to(pred.device)  # [3, 3, 33, 33, 33]
            # pred: [B, 3] → [B, 3, 1, 1, 1, 1]
            weights = pred.view(B, 3, 1, 1, 1, 1)
            # LUTs: [3, 3, D, D, D] → [1, 3, 3, D, D, D] → broadcast to [B, 3, 3, D, D, D]
            LUTs = LUTs.unsqueeze(0)  # shape [1, 3, 3, D, D, D]
            LUTs = LUTs.expand(B, -1, -1, -1, -1, -1)
            # LUT 가중합 계산: weighted sum over LUT axis (dim=1)
            # → 결과: [B, 3, 33, 33, 33]
            final_luts = (weights * LUTs).sum(dim=1)
            # wnorm 계산: 평균 제곱합 (스칼라)
            wnorm = pred.pow(2).mean()
            outputs = []

            # trilinear_.apply()가 batch를 지원하지 않는 경우: 하나씩 처리
            for i in range(B):
                lut = final_luts[i]  # [3, 33, 33, 33]
                input_img = img[i:i+1]  # [1, 3, H, W]
                out = trilinear_.apply(lut.unsqueeze(0), input_img)
                outputs.append(out)

            # 결과 병합
            return torch.cat(outputs, dim=0), wnorm

    def calculate_psnr():
        classifier.eval()
        model_rgbnir = RGBNIRtoRGB().to(device)
        psnr_sum = 0.0
        for batch in test_loader:
            rgb = batch['A_input'].to(device)
            nir = batch['A_NIR'].to(device)
            real_A = model_rgbnir(rgb,nir)
            real_A = real_A.type(Tensor)
            real_B = batch['A_target'].to(device).type(Tensor)
            fake_B,_ = generator_eval(real_A)
            w = min(fake_B.size(-1), real_B.size(-1))
            fake_B = fake_B[...,:w]; real_B = real_B[...,:w]
            mse = criterion((fake_B*255).round(), (real_B*255).round())
            psnr_sum += 10*math.log10((255.0**2)/mse.item())
        return psnr_sum/len(test_loader)

    max_psnr, max_epoch = 0.0, start_epoch
    prev_time = time.time()
    model_rgbnir = RGBNIRtoRGB().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5)

    
    for epoch in range(start_epoch, opt.n_epochs):
        classifier.train()
        mse_avg, psnr_acc = 0, 0
        for i, batch in enumerate(train_loader):
            rgb = batch['A_input'].to(device)
            nir = batch['A_NIR'].to(device)
            real_A = model_rgbnir(rgb,nir).type(Tensor)
            real_B = batch['A_target'].to(device).type(Tensor)
            optimizer_G.zero_grad()
            fake_B, wnorm = generator_train(real_A)
            mse = criterion(fake_B, real_B)
            tv = sum(TV3(lut)[0] for lut in [LUT0.LUT, LUT1.LUT, LUT2.LUT])
            mn = sum(TV3(lut)[1] for lut in [LUT0.LUT, LUT1.LUT, LUT2.LUT])

            loss = mse + opt.lambda_smooth*(wnorm+tv) + opt.lambda_monotonicity*mn
            scheduler.step(loss.item())
            loss.backward()
            optimizer_G.step()
            psnr_acc += 10 * math.log10(1 / mse.item())
            mse_avg += mse.item()
                           
            batches_done = epoch*len(train_loader) + i
            batches_left = opt.n_epochs*len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left*(time.time()-prev_time))
            prev_time = time.time()
            sys.stdout.write(
                f"\r[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(train_loader)}] "
                f"[psnr: {psnr_acc/(i+1):.6f}, tv: {tv:.6f}, "
                f"wnorm: {wnorm:.6f}, mn: {mn:.6f}] ETA: {time_left}"
            )
        avg_psnr = calculate_psnr()
        if avg_psnr > max_psnr: 
            max_psnr, max_epoch = avg_psnr, epoch
        sys.stdout.write(f" [PSNR: {avg_psnr:.6f}] [max PSNR: {max_psnr:.6f}, epoch: {max_epoch}]\n")

    if epoch % opt.checkpoint_interval == 0:
        LUTs = {str(i):lut.state_dict() for i,lut in enumerate([LUT0,LUT1,LUT2])}
        torch.save(LUTs, f"saved_models/{opt.output_dir}/LUTs_{epoch}.pth")
        torch.save(classifier.state_dict(), f"saved_models/{opt.output_dir}/classifier_{epoch}.pth")
        torch.save(optimizer_G.state_dict(), f"saved_models/{opt.output_dir}/optimizer_{epoch}.pth")
        with open(f"saved_models/{opt.output_dir}/result.txt","a") as f:
            f.write(f"[PSNR:{avg_psnr:.6f}] [max PSNR:{max_psnr:.6f}, epoch:{max_epoch}]\n")

if __name__ == "__main__":
    freeze_support()
    main()
