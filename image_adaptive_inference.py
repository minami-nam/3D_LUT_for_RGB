import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import *
from datasets import ImageDataset_sRGB



def main():
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=22)
    parser.add_argument("--dataset_name", type=str, default="fiveK")
    parser.add_argument("--model_dir", type=str, default="RGB_LUT_Classifier_Ckpt")
    parser.add_argument("--num_workers_eval", type=int, default=0)  # Windows면 0 권장
    parser.add_argument("--lambda_smooth", type=float, default=0.0001, help="smooth regularization")
    parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization")
    
    opt = parser.parse_args()

    def _to01(x: torch.Tensor) -> torch.Tensor:
    # Heuristic: if outside [0,1] significantly, assume [-1,1] and map to [0,1]
        if torch.min(x).item() < -0.05 or torch.max(x).item() > 1.05:
            return ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)



    # Cuda Device 지정 및 gradient 설정 금지 (inference이므로)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)  # 전체 스크립트 no-grad

    # ===== 1) LUT 및 Classifier 불러오기  =====
    LUT0 = Generator3DLUT_identity(dim=33).to(device)
    LUT1 = Generator3DLUT_zero().to(device)
    LUT2 = Generator3DLUT_zero().to(device)
    classifier = Classifier().to(device)
    TV3 = TV_3D().to(device)

    if device == "cuda":
        LUT0 = LUT0.cuda()
        LUT1 = LUT1.cuda()
        LUT2 = LUT2.cuda()
        classifier = classifier.cuda()
        TV3 = TV3.cuda()

    LUTs = torch.load(f"saved_models/{opt.model_dir}/LUTs_{opt.epoch}.pth")
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    classifier.load_state_dict(torch.load(f"saved_models/{opt.model_dir}/classifier_{opt.epoch}.pth"))


    # ===== 2) 데이터셋 =====
    root = f"data/{opt.dataset_name}"
    dataset = ImageDataset_sRGB(root, mode="test")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers_eval,
        pin_memory=torch.cuda.is_available(),
    )
    
    # ===== 3) 출력 폴더 =====
    out_dir = os.path.join("images_new", f"{opt.model_dir}_{opt.epoch}")
    os.makedirs(out_dir, exist_ok=True)
    # if opt.save_weights:
    #     os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)


    def gen_inference(img_prev):
        # Classifier를 통한 가중치 추출 + softmax
        classifier.eval()
        img = _to01(img_prev)
        pred = classifier(img)
        pred = F.softmax(pred, dim=1)  # [B, L]

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

    # ===== 4) 추론 루프 =====
    with torch.inference_mode():
        loss_sum, psnr_sum = 0.0, 0.0
        for batch in dataloader:
            img = batch["A_input"].to(device)# [B,3,H,W]
            real = batch["A_target"].to(device)
            name = batch["input_name"][0]
            base = os.path.splitext(name)[0] + ".png"

            # (1) LUT 통과 후 이미지 생성
            output = gen_inference(img)
            
            tv1, mn1 = TV3(LUT0)
            tv2, mn2 = TV3(LUT1)
            tv3, mn3 = TV3(LUT2) 
            
            tv = tv1 + tv2 + tv3
            mn = mn1 + mn2 + mn3
               
            mse_psnr = F.mse_loss(output, real)
            total_loss = mse_psnr + opt.lambda_smooth*tv + opt.lambda_monotonicity*mn
            
            # PSNR (per-batch)
            with torch.no_grad():
                psnr = 10.0 * torch.log10(1.0 / (mse_psnr + 1e-12))
                psnr_sum += psnr.item()
                loss_sum += total_loss.item()
            

            # (2) 저장
            save_image(output, os.path.join(out_dir, base), nrow=1, normalize=False)

            # # (3) (선택) LUT 가중치 저장
            # if args.save_weights and lut_weights is not None:
            #     # [B,num_LUTs,1,1,1] -> [num_LUTs]
            #     w = lut_weights[0, :, 0, 0, 0].detach().cpu()
            #     torch.save({"name": name, "weights": w}, os.path.join(out_dir, "weights", base.replace(".png", ".pt")))
    
            with open(f"{out_dir}/result.txt","a") as f:
                f.write(f"[PSNR:{psnr:.6f}] [loss:{total_loss}]\n")

    print(f"[+] Inference done. Results saved in {out_dir}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  
    main()

