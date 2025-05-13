import os

import gym
from PIL import Image, ImageDraw, ImageFont

# 保存用ディレクトリの作成
os.makedirs("frames_mountains", exist_ok=True)

env = gym.make("MountainCar-v0")

for episode in range(5):
    observation = env.reset()
    for step in range(100):
        # rgb_arrayモードでレンダリング
        rgb_array = env.render(mode="rgb_array")

        # PIL画像に変換
        img = Image.fromarray(rgb_array)

        # 描画オブジェクトを作成
        draw = ImageDraw.Draw(img)

        # デフォルトフォントを使用（サイズ20）
        # フォントがインストールされている場合は特定のフォントを指定することも可能
        try:
            # システムフォントがある場合
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except IOError:
            # デフォルトフォントを使用
            font = ImageFont.load_default()

        # テキストを描画（右上に配置）
        text = f"Episode: {episode} Step: {step}"
        text_width, text_height = (
            draw.textsize(text, font=font) if hasattr(draw, "textsize") else (300, 20)
        )
        position = (10, 10)  # 右上に配置するための座標

        # 背景を追加して読みやすくする
        draw.rectangle(
            [
                position[0] - 5,
                position[1] - 5,
                position[0] + text_width + 5,
                position[1] + text_height + 5,
            ],
            fill=(0, 0, 0, 128),  # 半透明の黒背景
        )

        # テキスト描画
        draw.text(position, text, font=font, fill=(255, 255, 255))  # 白色テキスト

        # 画像として保存
        img.save(f"frames_mountains/episode_{episode}_step_{step}.png")

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break

env.close()
