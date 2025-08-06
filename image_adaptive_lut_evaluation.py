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
    parser.add_argument("--epoch", type=int, default=140, help="epoch to load the saved checkpoint")
    parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
    parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
    parser.add_argument("--model_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10",
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
    # LUTs = torch.load(f"saved_models/{opt.model_dir}/LUTs_{opt.epoch}.pth")
    LUTs = torch.load(f"pretrained_models/sRGB/LUTs.pth")
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    LUT0.eval(); LUT1.eval(); LUT2.eval()

    # classifier.load_state_dict(torch.load(f"saved_models/{opt.model_dir}/classifier_{opt.epoch}.pth"))
    classifier.load_state_dict(torch.load(f"pretrained_models/sRGB/classifier.pth"))
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
        num_workers=0,    # <= Windows 에선 0으로
        pin_memory=cuda
    )

    # trilinear 적용 함수
    def generator(img):
        pred = classifier(img).squeeze()
        LUT = pred[0]*LUT0.LUT + pred[1]*LUT1.LUT + pred[2]*LUT2.LUT
        return TrilinearInterpolation.apply(LUT, img)

    # 결과 저장
    # out_dir = os.path.join("images", f"{opt.model_dir}_{opt.epoch}")
    out_dir = os.path.join("results", f"test_results")
    os.makedirs(out_dir, exist_ok=True)

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

