import os

import gym
from PIL import Image

# 保存用ディレクトリの作成
os.makedirs("frames", exist_ok=True)

env = gym.make("CartPole-v0")

for episode in range(5):
    observation = env.reset()
    for step in range(100):
        # rgb_arrayモードでレンダリング
        rgb_array = env.render(mode="rgb_array")

        # 画像として保存
        img = Image.fromarray(rgb_array)
        img.save(f"frames/episode_{episode}_step_{step}.png")

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break

env.close()
