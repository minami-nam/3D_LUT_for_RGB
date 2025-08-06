import shutil
from pathlib import Path
import random
import re

# Natural sort (e.g., a2.jpg < a10.jpg)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

# 가장 많은 공통된 24,916장의 파일 목록
input_list_path = Path("data/input_list.txt")
with input_list_path.open("r", encoding="utf-8") as f:
    valid_names = sorted(
        {line.strip() for line in f if line.strip()},
        key=natural_sort_key
    )

print(f"✅ Valid image count: {len(valid_names)}")

# 이미지 복사 함수
def filter_and_move_images(src_dir, dst_dir, valid_names):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for name in valid_names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            count += 1
    print(f"{src_dir.name}: {count} images moved → {dst_dir}")

# 이동 대상 폴더들
root = Path("data")
filter_and_move_images(root / "input_resized",  root / "fiveK/input/JPG", valid_names)
filter_and_move_images(root / "NIR_of_highRGB_block_resized", root / "fiveK/NIR/JPG", valid_names)
filter_and_move_images(root / "correct_resized", root / "fiveK/Target/JPG", valid_names)
filter_and_move_images(root / "test_resized", root / "fiveK/input/JPG", valid_names)
filter_and_move_images(root / "target_resized", root / "fiveK/Target/JPG", valid_names)

# Split train/test
random.seed(42)
random.shuffle(valid_names)

split_ratio = 0.9
n_train = int(len(valid_names) * split_ratio)
train_names = valid_names[:n_train]
test_names  = valid_names[n_train:]

# 리스트 저장 함수
def save_list(names, path):
    with open(path, "w", encoding="utf-8") as f:
        for name in sorted(names, key=natural_sort_key):
            f.write(name + "\n")
    print(f"📄 Saved: {path} ({len(names)} images)")

# 목록 저장
save_list(train_names, root / "input_list.txt")
save_list(train_names, root / "input_NIR_list.txt")
save_list(train_names, root / "correct.txt")
save_list(test_names, root / "test.txt")
save_list(test_names, root / "Target_list.txt")
