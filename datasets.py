import os
from pathlib import Path
import random
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision_x_functional as TF_x
import math

min_dist = 480

# ──────────────── 사전 공통 함수 ────────────────
def read_name_list(txt_path: Path):
    """TXT 한 줄에 하나씩, 파일명만 (확장자 포함) 리턴."""
    return [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def build_paths(root: Path, subdir: str, names: list):
    """root/subdir 에 name 을 붙여 Path 객체 리스트로 반환."""
    base = root / subdir
    paths = [base / name for name in names]
    # 존재 여부 체크
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
    return paths


# 2025. 08. 05 sRGB Paired만 수정함.

# ──────────────── sRGB Paired ────────────────
class ImageDataset_sRGB(Dataset):
    def __init__(self, root, mode="train"):
        self.root = Path(root)
        self.mode = mode

        # 읽을 TXT 파일 경로
        input_list = self.root / "input_list.txt"
        Target_list = self.root / "correct.txt"
        test_list  = self.root / "test.txt"

        # train 모드
        if mode == "train":
            names_inp = read_name_list(input_list)
            names_inp_tar = read_name_list(Target_list)
            self.input_files      = build_paths(self.root, "input/JPG/", names_inp)
            self.target_files     = build_paths(self.root, "Target/JPG", names_inp_tar)
        else:  # test 모드
            names_test    = read_name_list(test_list)
            self.input_files     = build_paths(self.root, "input/JPG", names_test)
            self.target_files    = build_paths(self.root, "inference_target/JPG", names_test)

        # transforms / to_tensor
        self.to_tensor = TF.to_tensor

        
    def __len__(self):
        return len(self.input_files)
    

    def __getitem__(self, idx):
        inp_path = self.input_files[idx]

        img_in  = Image.open(inp_path).convert("RGB")
        W, H = img_in.size

        if W >= H:
            H_new = min_dist
            W_new = int(round(W * H_new / H))
        else:
            W_new = min_dist
            H_new = int(round(H * W_new / W))

        img_in = img_in.resize((W_new, H_new))

        
        t_in  = self.to_tensor(img_in)   # [0,1]

        if self.mode == "train":
            tgt_path = self.target_files[idx]
            img_tgt = Image.open(tgt_path).convert("RGB")

            img_tgt = img_tgt.resize((W_new, H_new))
            
            t_tgt = self.to_tensor(img_tgt)

            return {
                "A_input":  t_in,
                "A_target": t_tgt,
                "input_name": inp_path.name
            }

        else:  
            tgt_path = self.target_files[idx]
            img_tgt = Image.open(tgt_path).convert("RGB")

            img_tgt = img_tgt.resize((W_new, H_new))
            
            t_tgt = self.to_tensor(img_tgt)# test
            return {
                "A_input":  t_in,
                "A_target": t_tgt,
                "input_name": inp_path.name
            }

