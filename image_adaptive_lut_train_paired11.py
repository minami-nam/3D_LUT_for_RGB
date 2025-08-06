import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys


from multiprocessing import freeze_support

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
    parser.add_argument("--n_epochs", type=int, default=200, help="total number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
    parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lambda_smooth", type=float, default=0.0001, help="smooth regularization")
    parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization")
    parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    parser.add_argument("--output_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10", help="path to save model")
    opt = parser.parse_args()

    opt.output_dir = opt.output_dir + '_' + opt.input_color_space
    print(opt)

    os.makedirs("saved_models/%s" % opt.output_dir, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion_pixelwise = torch.nn.MSELoss()

    LUT0 = Generator3DLUT_identity()
    LUT1 = Generator3DLUT_zero()
    LUT2 = Generator3DLUT_zero()
    classifier = Classifier()
    TV3 = TV_3D()
    trilinear_ = TrilinearInterpolation()

    if cuda:
        LUT0 = LUT0.cuda(); LUT1 = LUT1.cuda(); LUT2 = LUT2.cuda()
        classifier = classifier.cuda(); TV3 = TV3.cuda(); criterion_pixelwise.cuda()
        TV3.weight_r = TV3.weight_r.type(Tensor)
        TV3.weight_g = TV3.weight_g.type(Tensor)
        TV3.weight_b = TV3.weight_b.type(Tensor)

    if opt.epoch != 0:
        LUTs = torch.load(f"saved_models/{opt.output_dir}/LUTs_{opt.epoch}.pth")
        LUT0.load_state_dict(LUTs["0"])
        LUT1.load_state_dict(LUTs["1"])
        LUT2.load_state_dict(LUTs["2"])
        classifier.load_state_dict(torch.load(f"saved_models/{opt.output_dir}/classifier_{opt.epoch}.pth"))
    else:
        classifier.apply(weights_init_normal_classifier)
        torch.nn.init.constant_(classifier.model[16].bias.data, 1.0)

    optimizer_G = torch.optim.Adam(
        itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()),
        lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    # DataLoaders with num_workers=opt.n_cpu for training, 0 for testing
    if opt.input_color_space == 'sRGB':
        dataloader = DataLoader(
            ImageDataset_sRGB(f"data/{opt.dataset_name}", mode="train"),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            
        )
        psnr_dataloader = DataLoader(
            ImageDataset_sRGB(f"data/{opt.dataset_name}", mode="test"),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            
        )
    else:
        dataloader = DataLoader(
            ImageDataset_XYZ(f"data/{opt.dataset_name}", mode="train"),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
        psnr_dataloader = DataLoader(
            ImageDataset_XYZ(f"data/{opt.dataset_name}", mode="test"),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def generator_train(img):
        pred = classifier(img).squeeze()
        if len(pred.shape) == 1: pred = pred.unsqueeze(0)
        
        torch.cuda.synchronize()
        print("▶ before LUT0"); torch.cuda.synchronize()
        gen_A0 = LUT0(img)
        print("▶ after LUT0"); torch.cuda.synchronize()

        print("▶ before LUT1"); torch.cuda.synchronize()
        gen_A1 = LUT1(img)
        print("▶ after LUT1"); torch.cuda.synchronize()

        print("▶ before LUT2"); torch.cuda.synchronize()
        gen_A2 = LUT2(img)
        print("▶ after LUT2"); torch.cuda.synchronize()
        
        weights_norm = torch.mean(pred ** 2)
        combine_A = img.new_empty(img.size())
        for b in range(img.size(0)):
            combine_A[b] = (pred[b,0]*gen_A0[b] + pred[b,1]*gen_A1[b] + pred[b,2]*gen_A2[b])
        return combine_A, weights_norm

    def generator_eval(img):
        pred = classifier(img).squeeze()
        print("▶ before define LUT"); torch.cuda.synchronize()
        LUT = pred[0]*LUT0.LUT + pred[1]*LUT1.LUT + pred[2]*LUT2.LUT
        print("▶ after define LUT"); torch.cuda.synchronize()
        weights_norm = torch.mean(pred ** 2)
        combine_A = trilinear_.apply(LUT, img)
        return combine_A, weights_norm

    def calculate_psnr():
        model = RGBNIRtoRGB().to(device)
        classifier.eval()
        avg_psnr = 0
        for batch in psnr_dataloader:
            #
            rgb = batch["A_input"].to(device)
            nir = batch["A_NIR"].to(device)
            
            real_A = model(rgb, nir)
            real_A = real_A.type(Tensor)
            real_B = batch["A_target"].type(Tensor)
            fake_B, _ = generator_eval(real_A)
            # ———————————————— 크기 맞추기 ————————————————
            # fake_B 와 real_B 의 가로 픽셀 차이를 제거
            _,_,h1,w1 = fake_B.shape
            _,_,h2,w2 = real_B.shape
            # 높이는 같으니, 가로만 min(widths)
            w = min(w1, w2)
            fake_B = fake_B[:,:,:,:w]
            real_B = real_B[:,:,:,:w]

            mse = criterion_pixelwise(torch.round(fake_B*255), torch.round(real_B*255))
            psnr = 10 * math.log10(255.0**2 / mse.item())
            avg_psnr += psnr
        return avg_psnr / len(psnr_dataloader)

    # Training loop
    prev_time = time.time()
    max_psnr, max_epoch = 0, 0
    for epoch in range(opt.epoch, opt.n_epochs):
        classifier.train()
        mse_avg, psnr_acc = 0, 0
        model = RGBNIRtoRGB().to(device)
        for i, batch in enumerate(dataloader):
            rgb = batch["A_input"].to(device)
            nir = batch["A_NIR"].to(device)
            
            real_A = model(rgb, nir)
            real_A = real_A.type(Tensor)
            real_B = batch["A_target"].type(Tensor)
            optimizer_G.zero_grad()
            fake_B, weights_norm = generator_train(real_A)
            mse = criterion_pixelwise(fake_B, real_B)
            tv_cons = sum(TV3(lut)[0] for lut in [LUT0, LUT1, LUT2])
            mn_cons = sum(TV3(lut)[1] for lut in [LUT0, LUT1, LUT2])
            loss = mse + opt.lambda_smooth * (weights_norm + tv_cons) + opt.lambda_monotonicity * mn_cons
            psnr_acc += 10 * math.log10(1 / mse.item())
            mse_avg += mse.item()
            loss.backward(); optimizer_G.step()
            batches_done = epoch*len(dataloader) + i
            batches_left = opt.n_epochs*len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left*(time.time()-prev_time))
            prev_time = time.time()
            sys.stdout.write(
                f"\r[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                f"[psnr: {psnr_acc/(i+1):.6f}, tv: {tv_cons:.6f}, "
                f"wnorm: {weights_norm:.6f}, mn: {mn_cons:.6f}] ETA: {time_left}"
            )
        avg_psnr = calculate_psnr()
        if avg_psnr > max_psnr: max_psnr, max_epoch = avg_psnr, epoch
        sys.stdout.write(f" [PSNR: {avg_psnr:.6f}] [max PSNR: {max_psnr:.6f}, epoch: {max_epoch}]\n")

        if epoch % opt.checkpoint_interval == 0:
            LUTs = {str(i):lut.state_dict() for i,lut in enumerate([LUT0,LUT1,LUT2])}
            torch.save(LUTs, f"saved_models/{opt.output_dir}/LUTs_{epoch}.pth")
            torch.save(classifier.state_dict(), f"saved_models/{opt.output_dir}/classifier_{epoch}.pth")
            with open(f'saved_models/{opt.output_dir}/result.txt','a') as f:
                f.write(f" [PSNR: {avg_psnr:.6f}] [max PSNR: {max_psnr:.6f}, epoch: {max_epoch}]\n")

if __name__ == "__main__":
    freeze_support()
    main()
