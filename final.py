"""
통합 파이프라인 스크립트 (split 없이 전체 train + test 목록 생성):
1. input/target 폴더의 모든 이미지 파일을 높이 480px로 리사이즈 후 JPG로 변환
2. test_png 폴더의 PNG 파일을 높이 480px로 리사이즈 후 JPG로 변환
3. input_480p, target_480p, test_480p 폴더의 JPG 파일명을 모아 각각 train_input.txt, train_target.txt, test.txt 생성
"""
import os
from pathlib import Path
from PIL import Image

# RAW(DNG) 처리가 필요하면 rawpy 설치 후 활성화
try:
    import rawpy
except ImportError:
    rawpy = None
    print("⚠️ rawpy가 설치되어 있지 않습니다. DNG 파일은 건너뜁니다.")

# ———— 설정 ————
BASE_DIR = Path(__file__).resolve().parent / "data"
RAW_INPUT_DIR = BASE_DIR / "input"
RAW_NIR_DIR = BASE_DIR / "NIR_of_highRGB_block"
CORRECT_DIR = BASE_DIR / "correct"

TEST_RAW_DIR = BASE_DIR / "test"
TEST_TARGET_DIR = BASE_DIR / "target"

# ---- 출력 ----

RESIZED_INPUT_DIR = BASE_DIR / "input_resized"
RESIZED_NIR_DIR = BASE_DIR / "NIR_of_highRGB_block_resized"
RESIZED_CORRECT_DIR = BASE_DIR / "correct_resized"

TEST_RESIZED_DIR = BASE_DIR / "test_resized"
TEST_TARGET_RESIZED_DIR = BASE_DIR / "target_resized"

TARGET_HEIGHT = 480

# ——————————————————
def resize_images(src_dir: Path, dst_dir: Path, height: int = TARGET_HEIGHT):
    """
    src_dir 의 이미지들을 height 로 리사이즈 후 JPG로 dst_dir 에 저장
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    normal_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    raw_exts = {'.dng'}
    count = 0

    for img_path in sorted(src_dir.iterdir()):
        if not img_path.is_file():
            continue
        ext = img_path.suffix.lower()
        # RAW
        if ext in raw_exts:
            if rawpy is None:
                continue
            try:
                with rawpy.imread(str(img_path)) as raw:
                    img = Image.fromarray(raw.postprocess())
            except Exception as e:
                print(f"[Error] RAW 처리 실패: {img_path.name} ({e})")
                continue
        # 일반 이미지
        elif ext in normal_exts:
            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"[Error] 이미지 열기 실패: {img_path.name} ({e})")
                continue
        else:
            continue

        try:
            w, h = img.size
            new_w = int(w * height / h)
            img_resized = img.resize((new_w, height), Image.LANCZOS).convert("RGB")
            out_name = img_path.stem + ".jpg"
            img_resized.save(dst_dir / out_name, "JPEG")
            count += 1
        except Exception as e:
            print(f"[Error] 리사이즈 실패: {img_path.name} ({e})")

    print(f"{src_dir.name}: {count}개 이미지 리사이즈 완료 → {dst_dir.name}")


def generate_list(resized_dir: Path, list_path: Path):
    """
    resized_dir 내의 .jpg 파일명(확장자 포함)을 list_path에 기록
    """
    resized_dir = Path(resized_dir)
    files = sorted([f.name for f in resized_dir.iterdir()
                    if f.is_file() and f.suffix.lower() == '.jpg'])
    with open(list_path, 'w', encoding='utf-8') as f:
        for name in files:
            f.write(name + '\n')
    print(f"📋 목록 생성: {list_path.name} ({len(files)}개)")


if __name__ == '__main__':
    # 1) 이미지 리사이즈
    print('1) Train input 이미지 리사이즈 중…')
    resize_images(RAW_INPUT_DIR, RESIZED_INPUT_DIR)
    print('2) NIR 이미지 리사이즈 중…')
    resize_images(RAW_NIR_DIR, RESIZED_NIR_DIR)
    print('3) 정답 이미지 리사이즈 중…')
    resize_images(CORRECT_DIR, RESIZED_CORRECT_DIR)
    print('4) test 이미지 리사이즈 중...')
    resize_images(TEST_RAW_DIR, TEST_RESIZED_DIR)
    print('5) Target 이미지 리사이즈 중...')
    resize_images(TEST_TARGET_DIR, TEST_TARGET_RESIZED_DIR)
    

    # 2) 목록 생성
    print('6) input_list.txt 생성…')
    generate_list(RESIZED_INPUT_DIR, BASE_DIR / 'input_list.txt')
    print('7) input_NIR_list.txt 생성…')
    generate_list(RESIZED_NIR_DIR, BASE_DIR / 'input_NIR_list.txt')
    print('8) correct.txt 생성… 학습 시 파일의 이름을 Target_list.txt로 바꾼 후 사용하세요.')
    generate_list(RESIZED_CORRECT_DIR, BASE_DIR / 'correct.txt')
    print('9) test.txt 생성…')
    generate_list(TEST_RESIZED_DIR, BASE_DIR / 'test.txt')
    print('10) Target_list.txt 생성…')
    generate_list(TEST_TARGET_RESIZED_DIR, BASE_DIR / 'Target_list.txt')
    
    