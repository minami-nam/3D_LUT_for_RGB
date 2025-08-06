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
        """
        root/ 에 txt 파일이 생성되어 있어야합니다.
        """
        self.root = Path(root)
        self.mode = mode

        # 읽을 TXT 파일 경로
        input_list = self.root / "input_list.txt"
        input_NIR_list = self.root / "input_NIR_list.txt"
        Target_list = self.root / "Target_list.txt"
        test_list  = self.root / "test.txt"

        # train 모드에서만 NIR
        if mode == "train":
            names_inp = read_name_list(input_list)
            names_inp_nir = read_name_list(input_NIR_list)
            names_inp_tar = read_name_list(Target_list)
            self.input_files  = build_paths(self.root, "input/JPG/",   names_inp)
            self.input_NIR_files = build_paths(self.root, "NIR/JPG", names_inp_nir)
            self.target_files = build_paths(self.root, "Target/JPG", names_inp_tar)
        else:
            names_test = read_name_list(test_list)
            self.input_files  = build_paths(self.root, "input/JPG",   names_test)
            self.target_files = build_paths(self.root, "Target/JPG", names_test)

        # transforms
        self.to_tensor = TF.to_tensor
        self.transforms = transforms.RandomCrop

    def __len__(self):
        return len(self.input_files)
    

    def __getitem__(self, idx):
        inp_path = self.input_files[idx]
        tgt_path = self.target_files[idx]

        img_in = Image.open(inp_path).convert("RGB")  # 
        img_tgt = Image.open(tgt_path).convert("RGB")
        t_in = self.to_tensor(img_in)
        t_tgt = self.to_tensor(img_tgt)
        
        if self.mode == "train":
            t_nir = self.to_tensor(Image.open(self.input_NIR_files[idx]).convert("L") )
        else:
            t_nir = torch.zeros_like(t_in[:1]) 
            
        
    
        return {
            "A_input" :  t_in,
            "A_NIR" : t_nir,
            "A_target" : t_tgt,
            "input_name": inp_path.name
        }

# ──────────────── XYZ Paired ────────────────
class ImageDataset_XYZ(Dataset):
    def __init__(self, root, mode="train"):
        self.root = Path(root)
        self.mode = mode

        input_list = self.root / "train_input.txt"
        label_list = self.root / "train_label.txt"
        test_list  = self.root / "test.txt"

        if mode == "train":
            names_inp = read_name_list(input_list)
            names_lbl = read_name_list(label_list)
            self.input_files  = build_paths(self.root, "input/PNG/480p_16bits_XYZ_WB", names_inp)
            self.expert_files = build_paths(self.root, "expertC/JPG/480p",            names_lbl)
        else:
            names_test = read_name_list(test_list)
            self.input_files  = build_paths(self.root, "input/PNG/480p_16bits_XYZ_WB", names_test)
            self.expert_files = build_paths(self.root, "expertC/JPG/480p",            names_test)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp_path = self.input_files[idx]
        tgt_path = self.expert_files[idx]

        # 16bit PNG은 cv2로 읽고, ExpertC JPG는 PIL
        img_in = cv2.imread(str(inp_path), -1)
        img_ex = Image.open(tgt_path).convert("RGB")

        if self.mode == "train":
            # 비슷한 augmentation 적용
            W, H = img_ex.size
            rH, rW = np.random.uniform(0.6,1.0), np.random.uniform(0.6,1.0)
            crop_h, crop_w = int(H*rH), int(W*rW)
            i, j, h, w = transforms.RandomCrop.get_params(img_ex, (crop_h, crop_w))
            img_in = TF_x.crop(img_in, i, j, h, w)
            img_ex = TF.crop(img_ex, i, j, h, w)
            if random.random() > 0.5:
                img_in = TF_x.hflip(img_in)
                img_ex = TF.hflip(img_ex)
            img_in = TF_x.adjust_brightness(img_in, np.random.uniform(0.6,1.4))

        return {
            "A_input":  TF_x.to_tensor(img_in),
            "A_exptC":  TF.to_tensor(img_ex),
            "input_name": inp_path.name
        }

# ──────────────── sRGB Unpaired ────────────────
class ImageDataset_sRGB_unpaired(Dataset):
    def __init__(self, root, mode="train"):
        self.root = Path(root)
        self.mode = mode

        input_list = self.root / "train_input.txt"
        label_list = self.root / "train_label.txt"
        test_list  = self.root / "test.txt"

        names1 = read_name_list(input_list)
        names2 = read_name_list(label_list)
        names_test = read_name_list(test_list)

        self.input_files  = build_paths(self.root, "input/JPG/480p",   names1 if mode=="train" else names_test)
        self.expert_files = build_paths(self.root, "expertC/JPG/480p", names1 if mode=="train" else names_test)
        # unpaired용 second expert list
        if mode=="train":
            self.other_expert = build_paths(self.root, "expertC/JPG/480p", names2)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp_path = self.input_files[idx]
        ex1_path = self.expert_files[idx]
        img_in = Image.open(inp_path).convert("RGB")
        img_ex = Image.open(ex1_path).convert("RGB")

        if self.mode=="train":
            # 랜덤으로 다른 expert 이미지 섞기
            ex2_path = random.choice(self.other_expert)
            img_ex2 = Image.open(ex2_path).convert("RGB")
        else:
            img_ex2 = img_ex

        # (여기에 augmentation 적용 가능)

        return {
            "A_input":  TF.to_tensor(img_in),
            "A_exptC":  TF.to_tensor(img_ex),
            "B_exptC":  TF.to_tensor(img_ex2),
            "input_name": inp_path.name
        }

# ──────────────── XYZ Unpaired ────────────────
class ImageDataset_XYZ_unpaired(Dataset):
    def __init__(self, root, mode="train"):
        self.root = Path(root)
        self.mode = mode

        input_list = self.root / "train_input.txt"
        label_list = self.root / "train_label.txt"
        test_list  = self.root / "test.txt"

        names1 = read_name_list(input_list)
        names2 = read_name_list(label_list)
        names_test = read_name_list(test_list)

        self.input_files  = build_paths(self.root, "input/PNG/480p_16bits_XYZ_WB", names1 if mode=="train" else names_test)
        self.expert_files = build_paths(self.root, "expertC/JPG/480p",            names1 if mode=="train" else names_test)
        if mode=="train":
            self.other_expert = build_paths(self.root, "expertC/JPG/480p", names2)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp_path = self.input_files[idx]
        ex1_path = self.expert_files[idx]
        img_in = cv2.imread(str(inp_path), -1)
        img_ex = Image.open(ex1_path).convert("RGB")

        if self.mode=="train":
            ex2_path = random.choice(self.other_expert)
            img_ex2 = Image.open(ex2_path).convert("RGB")
        else:
            img_ex2 = img_ex

        return {
            "A_input":  TF_x.to_tensor(img_in),
            "A_exptC":  TF.to_tensor(img_ex),
            "B_exptC":  TF.to_tensor(img_ex2),
            "input_name": inp_path.name
        }

# ──────────────── HDRplus ────────────────
class ImageDataset_HDRplus(Dataset):
    def __init__(self, root, mode="train"):
        self.root = Path(root)
        self.mode = mode

        train_list = self.root / "train.txt"
        test_list  = self.root / "test.txt"
        names = read_name_list(train_list if mode=="train" else test_list)

        sub_in  = "middle_480p"
        sub_out = "output_480p"

        self.input_files  = build_paths(self.root, sub_in,  names)
        self.expert_files = build_paths(self.root, sub_out, names)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp_path = self.input_files[idx]
        ex_path  = self.expert_files[idx]
        img_in = cv2.imread(str(inp_path), -1)
        img_ex = Image.open(ex_path).convert("RGB")

        return {
            "A_input":  TF_x.to_tensor(img_in),
            "A_exptC":  TF.to_tensor(img_ex),
            "input_name": inp_path.name
        }

# ──────────────── HDRplus Unpaired ────────────────
class ImageDataset_HDRplus_unpaired(Dataset):
    def __init__(self, root, mode="train"):
        self.root = Path(root)
        self.mode = mode

        train_list = self.root / "train.txt"
        test_list  = self.root / "test.txt"
        names1 = read_name_list(train_list)
        names2 = names1
        names_test = read_name_list(test_list)

        sub_in  = "middle_480p"
        sub_out = "output_480p"

        if mode=="train":
            self.input_files  = build_paths(self.root, sub_in, names1)
            self.expert_files = build_paths(self.root, sub_out, names1)
            self.other_expert = build_paths(self.root, sub_out, names2)
        else:
            self.input_files  = build_paths(self.root, sub_in, names_test)
            self.expert_files = build_paths(self.root, sub_out, names_test)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp_path = self.input_files[idx]
        img_in  = cv2.imread(str(inp_path), -1)
        img_ex1 = Image.open(self.expert_files[idx]).convert("RGB")
        if self.mode=="train":
            img_ex2 = Image.open(random.choice(self.other_expert)).convert("RGB")
        else:
            img_ex2 = img_ex1

        return {
            "A_input":  TF_x.to_tensor(img_in),
            "A_exptC":  TF.to_tensor(img_ex1),
            "B_exptC":  TF.to_tensor(img_ex2),
            "input_name": inp_path.name
        }
