import argparse
import os
import time

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import *
from datasets import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=40, help="epoch to load the saved checkpoint")
    parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
    parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
    parser.add_argument("--model_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT",
                        help="directory of saved models")
    opt = parser.parse_args()
    opt.model_dir = opt.model_dir + '_' + opt.input_color_space

    # GPU 사용 설정
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # 손실함수 및 모델 정의
    LUT0 = Generator3DLUT_identity().cuda() if cuda else Generator3DLUT_identity()
    LUT1 = Generator3DLUT_zero()    .cuda() if cuda else Generator3DLUT_zero()
    LUT2 = Generator3DLUT_zero()    .cuda() if cuda else Generator3DLUT_zero()
    classifier = Classifier().cuda() if cuda else Classifier()

    # 체크포인트 로드
    LUTs = torch.load(f"saved_models/{opt.model_dir}/LUTs_{opt.epoch}.pth")
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    LUT0.eval(); LUT1.eval(); LUT2.eval()

    classifier.load_state_dict(torch.load(f"saved_models/{opt.model_dir}/classifier_{opt.epoch}.pth"))
    classifier.eval()

    # DataLoader: 윈도우용으로 num_workers=0
    root = f"data/{opt.dataset_name}"
    if opt.input_color_space.lower() == 'srgb':
        dataset = ImageDataset_sRGB(root, mode="test")
    else:
        dataset = ImageDataset_XYZ(root, mode="test")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,    # <= Windows 에선 0으로
        pin_memory=cuda
    )

    # trilinear 적용 함수
    def generator(img):
        pred = classifier(img).squeeze()

        # pred.shape could be [3] or [B, 3]
        if pred.ndim == 1 and pred.shape[0] == 3:
            pred = pred.unsqueeze(0)  # [1, 3]
        
        B = pred.shape[0]

        # Expand LUTs
        lut0 = LUT0.LUT.unsqueeze(0).expand(B, -1, -1, -1, -1)
        lut1 = LUT1.LUT.unsqueeze(0).expand(B, -1, -1, -1, -1)
        lut2 = LUT2.LUT.unsqueeze(0).expand(B, -1, -1, -1, -1)

        # Expand weights
        weights = pred.view(B, 3, 1, 1, 1)

        # Weighted sum
        LUT = weights[:, 0:1] * lut0 + weights[:, 1:2] * lut1 + weights[:, 2:3] * lut2

        # Apply LUT transform
        fake_B = TrilinearInterpolation.apply(LUT.unsqueeze(0), img)
        return fake_B

        #LUT = pred[0]*LUT0.LUT + pred[1]*LUT1.LUT + pred[2]*LUT2.LUT
        #return TrilinearInterpolation.apply(LUT, img)

    # 결과 저장
    out_dir = os.path.join("images", f"{opt.model_dir}_{opt.epoch}")
    os.makedirs(out_dir, exist_ok=True)
    
    
    with torch.no_grad():
        for batch in dataloader:
            real_A = Variable(batch["A_input"].type(Tensor))
            img_name = batch["input_name"][0]                # "a0001-dgw_005.tif" 등
            fake_B = generator(real_A)
            fname = os.path.splitext(img_name)[0] + ".png"   # 확장자 제거 + .png
            save_image(fake_B, os.path.join(out_dir, fname), nrow=1, normalize=False)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

