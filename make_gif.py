import glob
import os
import re

from PIL import Image

# フレーム画像を保存したディレクトリ
frames_dir = "final_success"  # または "frames" に変更

# フレーム画像を取得してソート
frame_files = glob.glob(f"{frames_dir}/episode_*_step_*.png")


# 数字を正しくソートするための関数
def natural_sort_key(s):
    # 数字部分を数値として扱うための正規表現パターン
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


# 自然順（数値順）でソート
frames = sorted(frame_files, key=natural_sort_key)


# フレームを読み込む
images = []
for frame in frames:
    img = Image.open(frame)
    images.append(img)

# GIFとして保存（100ミリ秒間隔）
os.makedirs("animations", exist_ok=True)
images[0].save(
    f"animations/{frames_dir}.gif",
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=100,
    loop=0,
)

print("GIFアニメーションが作成されました: animations/cartpole.gif")
