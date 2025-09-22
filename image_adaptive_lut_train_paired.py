import argparse, os, time, math, datetime, sys
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import freeze_support
# import kornia as K
from models import *
from datasets import ImageDataset_sRGB

# from torch.optim.lr_scheduler import ReduceLROnPlateau

# --------------------- Cuda Device 선택  ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=102, help="start epoch")
    parser.add_argument("--n_epochs", type=int, default=400, help="total epochs")
    parser.add_argument("--dataset_name", type=str, default="fiveK", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--n_cpu", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="RGB_LUT_Classifier_Ckpt")
    parser.add_argument("--lambda_ssim", type=float, default=5)
    parser.add_argument("--lambda_mse", type=float, default = 10)
    parser.add_argument("--lambda_tv", type=float, default=0.01)
    parser.add_argument("--lambda_color", type=float, default=1, help="CIEDE2000 Loss")
    parser.add_argument("--lambda_smooth", type=float, default=0.01, help="smooth regularization")
    parser.add_argument("--lambda_monotonicity", type=float, default=0.01, help="monotonicity regularization")
    parser.add_argument("--norm", type=str, default="instance", choices=["instance","batch","none"], help="normalization type used in Classifier")
    opt = parser.parse_args()

    os.makedirs(f"saved_models/{opt.output_dir}", exist_ok=True)
    DEBUG_DIR = "train_samples"
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # --------------------- 정규화 함수  ---------------------
    def _to01(x: torch.Tensor) -> torch.Tensor:
        # Heuristic: if outside [0,1] significantly, assume [-1,1] and map to [0,1]
        if torch.min(x).item() < -0.05 or torch.max(x).item() > 1.05:
            return ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)


    # ----- sRGB <-> linear -----
    def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
        a = 0.055
        x = x.clamp(0.0, 1.0)
        return torch.where(x <= 0.04045, x/12.92, ((x + a)/1.055) ** 2.4)

    def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
        a = 0.055
        x = x.clamp(0.0, 1.0)
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 1/2.4) - a)

    # ----- linear RGB(NCHW) -> XYZ(NCHW) (sRGB, D65) -----
    def rgb_to_xyz_linear(x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] linear RGB in [0,1]
        returns XYZ: [B,3,H,W]
        """
        M = x.new_tensor([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])     # sRGB->XYZ(D65)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)             # [N,3]
        xyz = x_flat @ M.t()    # [N,3] 
        return xyz.reshape(B, H, W, 3).permute(0, 3, 1, 2)      # [B,3,H,W]

    # ----- XYZ(NCHW) -> Lab(NCHW) -----
    def xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [B,3,H,W], same scale as white (here 0..1)
        returns Lab: [B,3,H,W] with L in [0,100] approx
        """
        X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        Xn, Yn, Zn = xyz.new_tensor(0.95047), xyz.new_tensor(1.0), xyz.new_tensor(1.08883)  # D65
        xr, yr, zr = ((X / Xn)+5e-4), ((Y / Yn)+5e-4), ((Z / Zn)+5e-4)

        delta = 6/29
        th = delta**3
        def f(t):  # piecewise cube-root
            return torch.where(t > th, t.pow(1/3), t/(3*delta*delta) + 4/29)

        fx, fy, fz = f(xr), f(yr), f(zr)
        L = 116*fy - 16
        a = 500*(fx - fy)
        b = 200*(fy - fz)
        return torch.stack([L, a, b], dim=1)
    
    def torch_ciede2000_loss_stable(lab_true, lab_pred):
        # Pytorch는 constant와 비슷한 역할을 하는 함수가 없음.
        PI = np.pi
        TWO_PI = 2.0*np.pi
        EPS = 1e-8
        BIG_EPS = 1e-6 # 분모/루트용 좀 더 크게

        if lab_true.dim() == 4 and lab_true.size(1) == 3:   # NCHW
            L1,a1,b1 = lab_true[:,0], lab_true[:,1], lab_true[:,2]
            L2,a2,b2 = lab_pred[:,0], lab_pred[:,1], lab_pred[:,2]
        elif lab_true.shape[-1] == 3:                   # ...C last
            L1,a1,b1 = lab_true[...,0], lab_true[...,1], lab_true[...,2]
            L2,a2,b2 = lab_pred[...,0], lab_pred[...,1], lab_pred[...,2]
        else:
            raise ValueError("lab tensors must have a size-3 channel")

        # (옵션) Lab 예측값을 물리적 범위로 부드럽게 제한: tanh 스케일링
        # 완전 hard-clip은 비미분점이 생기므로 tanh 권장
        # L2 = 50.0*(torch.tanh(L2/50.0))+50.0         # 대략 [0,100] 근처
        # a2 = 128.0*torch.tanh(a2/128.0)              # 대략 [-128,128]
        # b2 = 128.0*torch.tanh(b2/128.0)
    

        # 1) 기본 Chroma
        C1 = torch.sqrt(a1*a1 + b1*b1 + EPS)
        C2 = torch.sqrt(a2*a2 + b2*b2 + EPS)
        C_bar = 0.5*(C1 + C2)
        # 2) pow7 안정화: base를 먼저 제한해 overflow 자체를 방지
        # float32에서 (1e30)^(1/7) ≈ 2e4 정도. 여유롭게 1e4~2e4 사이로 제한.
        POW7_BASE_LIMIT = 2e4
        def pow7_safe(x):
            x_clip = torch.minimum(torch.maximum(x, torch.zeros_like(x, device=device)), POW7_BASE_LIMIT * torch.ones_like(x, device=device))
            return torch.pow(x_clip, 7.0)

        num = pow7_safe(C_bar)
        den = num + 25.0**7
        # 분수는 항상 [0,1]로 clamp
        frac = num / torch.maximum(den, BIG_EPS*torch.ones_like(den, device=device))
        frac = torch.clamp(frac, 0.0, 1.0)

        G = 0.5 * (1.0 - torch.sqrt(frac))

        a1p = (1.0 + G) * a1
        a2p = (1.0 + G) * a2
        C1p = torch.sqrt(a1p*a1p + b1*b1 + EPS)
        C2p = torch.sqrt(a2p*a2p + b2*b2 + EPS)

        # C'≈0이면 hue는 정의되지 않으므로 표준 특례: h'을 0으로 두고 이후 항이 영향 거의 없도록 함
        C1p_nz = C1p > BIG_EPS
        C2p_nz = C2p > BIG_EPS

        h1p_raw = torch.atan2(b1, a1p + EPS)  # 분모 EPS
        h2p_raw = torch.atan2(b2, a2p + EPS)
        h1p_raw = torch.where(h1p_raw < 0, h1p_raw + TWO_PI, h1p_raw)
        h2p_raw = torch.where(h2p_raw < 0, h2p_raw + TWO_PI, h2p_raw)

        h1p = torch.where(C1p_nz, h1p_raw, torch.zeros_like(h1p_raw, device=device))
        h2p = torch.where(C2p_nz, h2p_raw, torch.zeros_like(h2p_raw, device=device))

        # 3) deltas
        dLp = L2 - L1
        dCp = C2p - C1p

        hdiff = h2p - h1p
        hdiff = torch.where(hdiff >  PI, hdiff - TWO_PI, hdiff)
        hdiff = torch.where(hdiff < -PI, hdiff + TWO_PI, hdiff)

        # mult = sqrt(C1p*C2p) -> 0에서 기울기 발산 방지
        mult = torch.sqrt(torch.maximum(C1p*C2p, BIG_EPS*torch.ones_like(C1p*C2p)))
        dHp = 2.0 * mult * torch.sin(0.5 * hdiff)
        # 표준 특례: 둘 중 하나라도 C'≈0이면 dH'≈0
        dHp = torch.where(C1p_nz & C2p_nz, dHp, torch.zeros_like(dHp, device=device))

        # 4) 가중치/회전항
        Lbp = 0.5 * (L1 + L2)
        Cbp = 0.5 * (C1p + C2p)

        # hue 평균도 특례 필요: 두 쪽 모두 C' > 0일 때만 의미
        hb_same = torch.abs(h1p - h2p) <= PI
        hbp_raw = torch.where(hb_same, 0.5*(h1p + h2p), 0.5*(h1p + h2p + TWO_PI))
        hbp = torch.where(C1p_nz & C2p_nz, hbp_raw, BIG_EPS*torch.ones_like(hbp_raw, device=device))

        # T
        T = (1.0
            - 0.17 * torch.cos(hbp - PI/6.0)
            + 0.24 * torch.cos(2.0*hbp)
            + 0.32 * torch.cos(3.0*hbp + PI/30.0)
            - 0.20 * torch.cos(4.0*hbp - 63.0*np.pi/180.0))
        T = torch.clamp(T, -2.0, 2.0)

        delta_theta = 30.0*np.pi/180.0 * torch.exp(-torch.square((hbp*180.0/PI - 275.0) / 25.0))

        RC_num = pow7_safe(Cbp)
        RC = 2.0 * torch.sqrt(torch.clamp(RC_num / torch.maximum(RC_num + 25.0**7, BIG_EPS*torch.ones_like(RC_num)), 0.0, 1.0))
        RT = -torch.sin(2.0 * delta_theta) * RC
        RT = torch.clamp(RT, -1.5, 1.5)

        SL = 1.0 + (0.015 * torch.square(Lbp - 50.0)) / torch.sqrt(20.0 + torch.square(Lbp - 50.0))
        SC = 1.0 + 0.045 * Cbp
        SH = 1.0 + 0.015 * Cbp * T
        SL = torch.maximum(SL, BIG_EPS*torch.ones_like(SL)); SC = torch.maximum(SC, BIG_EPS*torch.ones_like(SC)); SH = torch.maximum(SH, BIG_EPS*torch.ones_like(SH))

        kL = kC = kH = 1.0
        termL = torch.square(dLp / (kL * SL))
        termC = torch.square(dCp / (kC * SC))
        termH = torch.square(dHp / (kH * SH))

        cross = RT * (dCp / (kC * SC)) * (dHp / (kH * SH)) + 1e-9
        # cross도 폭이 크면 clamp
        cross = torch.clamp(cross, -1e6, 1e6)

        rad = termL + termC + termH + cross

        # (선택 1) sqrt를 쓰면 0 근처 gradient가 날카롭습니다 → EPS 추가
        # dE = tf.sqrt(tf.maximum(rad, 0.0) + BIG_EPS)

        # (선택 2, 추천) deltaE^2를 loss로 사용 (미분 더 안정)
        dE2 = torch.maximum(rad, torch.zeros_like(rad, device=device))

        # forward finite 체크 (학습 중엔 끄는 것도 고려)
        # tf.debugging.assert_all_finite(dE2, "deltaE2000^2")
        dE2 = torch.sqrt(dE2)

        # 평균 반환
        return torch.mean(dE2)  # 또는 tf.reduce_mean(dE) if sqrt 버전 사용


    # ----- Convenience: ΔE00 directly from sRGB tensors -----
    def delta_e00_from_srgb(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        img1, img2: sRGB in [0,1], shape [B,3,H,W] (NCHW)
        returns ΔE00: [B,H,W]
        """
        lin1 = srgb_to_linear(img1)
        lin2 = srgb_to_linear(img2)
        xyz1 = rgb_to_xyz_linear(lin1)
        xyz2 = rgb_to_xyz_linear(lin2)
        lab1 = xyz_to_lab(xyz1)       # [B,3,H,W]
        lab2 = xyz_to_lab(xyz2)
        return torch_ciede2000_loss_stable(lab1, lab2)  # [B,H,W]
    


    
    # --- Data loader ---
    Dataset = ImageDataset_sRGB 
    train_dataset = Dataset(f"data/{opt.dataset_name}", mode="train")


    # persistent_workers는 num_workers>0일 때만 안전
    use_pw = opt.n_cpu > 0
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.n_cpu, persistent_workers=use_pw)
    
    LUT0 = Generator3DLUT_identity(dim=33).to(device)
    LUT1 = Generator3DLUT_zero().to(device)
    LUT2 = Generator3DLUT_zero().to(device)
    classifier = Classifier().to(device)
    TV3 = TV_3D().to(device)


    # --- Model & Optimizer ---
    optimizer_G = torch.optim.Adam(itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()),
        lr=opt.lr, betas=(opt.b1, opt.b2)
    )


    
    #  --------------------- 평가 PSNR  ---------------------
    @torch.no_grad()
    def evaluate_psnr(forward, data_loader: DataLoader, max_batches: int = 16) -> float:
        classifier.eval()
        psnr_sum, count = 0.0, 0
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
            input = batch['A_input'].to(device)
            real = batch['A_target'].to(device)

            fake = forward(input)
            fake01 = _to01(fake)
            real01 = _to01(real)
            # MSE on [0,1]
            mse = F.mse_loss(fake01, real01)
            # Avoid log of zero
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
            psnr_sum += psnr.item()
            count += 1
        return psnr_sum / max(1, count)

    # --------------------- Training 관련 ------------------------
    def gen_train(img_prev):
        # Classifier를 통한 가중치 추출 + softmax
        img = _to01(img_prev)
        pred = classifier(img)
        pred = F.softmax(pred, dim=1)

        # LUT에 이미지를 넣고 결과들을 추출하여 stack
        c0 = LUT0(img) # [B, 3, H, W]로 Tensor 구성
        c1 = LUT1(img)
        c2 = LUT2(img)
        

        cout = [c0, c1, c2]
        LUT_all = torch.stack(cout, dim=1) # [B, L, 3, H, W] 여기서 L은 3

        # B와 L을 Define 하기
        B, C, H, W = img.shape
        L = LUT_all.size(1) # LUT 개수

        # 가중합 처리
        out = (LUT_all * pred.view(B, L, 1, 1, 1)).sum(dim=1)

        return out

    start_epoch = 0
    if opt.epoch != 0:
        LUTs = torch.load(f"saved_models/{opt.output_dir}/LUTs_{opt.epoch}.pth")
        LUT0.load_state_dict(LUTs["0"])
        LUT1.load_state_dict(LUTs["1"])
        LUT2.load_state_dict(LUTs["2"])
        classifier.load_state_dict(torch.load(f"saved_models/{opt.output_dir}/classifier_{opt.epoch}.pth"))
        opt_state = torch.load(f"saved_models/{opt.output_dir}/optimizer_{opt.epoch}.pth", map_location=device)
        optimizer_G.load_state_dict(opt_state)
        start_epoch = opt.epoch
    else:
        classifier.apply(weights_init_normal_classifier)
        torch.nn.init.constant_(classifier.model[16].bias.data, 1.0)

    prev_time = time.time()

    for epoch in range(start_epoch, opt.n_epochs):
        classifier.train()
        loss_sum, psnr_sum = 0.0, 0.0

        for i, batch in enumerate(train_loader):
            rgb  = batch['A_input'].to(device)   # [B,3,H,W]
            real = batch['A_target'].to(device)  # [B,3,H,W]

            optimizer_G.zero_grad()

            # Forward
            fake = gen_train(rgb)     # fake: [B,3,H,W]    
            tv1, mn1 = TV3(LUT0)
            tv2, mn2 = TV3(LUT1)
            tv3, mn3 = TV3(LUT2) 
            
            tv = tv1 + tv2 + tv3
            mn = mn1 + mn2 + mn3
            

            ssim = SSIM().to(device)
            ssim_loss = ssim(_to01(fake),_to01(real)).mean()
            ciede2000_loss = delta_e00_from_srgb(_to01(fake),_to01(real)).mean()


            mse_psnr = F.mse_loss(_to01(fake), _to01(real))
            # total_var_loss = total_variation_loss(fake, opt.lambda_tv)
            # color_loss = color_loss_e(fake, real)
            total_loss = opt.lambda_mse*mse_psnr + opt.lambda_smooth*tv + opt.lambda_monotonicity*mn + ssim_loss*opt.lambda_ssim + opt.lambda_color*ciede2000_loss

            # Backprop
            total_loss.backward()
            optimizer_G.step()

            # PSNR (per-batch)
            with torch.no_grad():
                psnr = 10.0 * torch.log10(1.0 / (mse_psnr + 1e-12))
                psnr_sum += psnr.item()
                loss_sum += total_loss.item()

            # ETA 표시
            batches_done = epoch * len(train_loader) + i + 1
            batches_left = opt.n_epochs * len(train_loader) - batches_done
            iter_time = time.time() - prev_time
            prev_time = time.time()
            eta = datetime.timedelta(seconds=int(batches_left * max(iter_time, 1e-9)))


            sys.stdout.write(
                f"\r[Epoch {epoch}/{opt.n_epochs}] [Batch {i+1}/{len(train_loader)}] "
                f"[psnr: {psnr_sum/(i+1):.6f}, loss: {total_loss.item():.6f}] "
                f"[mse:{mse_psnr.item():.6f}] ETA: {eta}"
            )
        # tv:{total_var_loss.item():.6f}, color:{color_loss.item():.6f}

        # --- Validation (옵션: 간단히 train psnr 평균만 기록) ---
        avg_psnr = psnr_sum / max(1, len(train_loader))
        # Proper eval-mode PSNR using running stats (for BN) and without dropout
        eval_psnr = evaluate_psnr(gen_train, train_loader, max_batches=8)
        print(f"\n[Epoch {epoch}] train_psnr(mean): {avg_psnr:.6f} | eval_psnr: {eval_psnr:.6f} | total_loss:{total_loss:.6f}")

        # Save checkpoint

        if epoch > 0:
            LUTs = {str(i):lut.state_dict() for i,lut in enumerate([LUT0,LUT1,LUT2])}
            torch.save(LUTs, f"saved_models/{opt.output_dir}/LUTs_{epoch}.pth")
            torch.save(classifier.state_dict(), f"saved_models/{opt.output_dir}/classifier_{epoch}.pth")
            torch.save(optimizer_G.state_dict(), f"saved_models/{opt.output_dir}/optimizer_{epoch}.pth")
            with open(f"saved_models/{opt.output_dir}/result.txt","a") as f:
                f.write(f"[PSNR:{avg_psnr:.6f}] [max PSNR:{eval_psnr:.6f}, epoch:{total_loss}]\n")


if __name__ == "__main__":
    freeze_support()
    main()
