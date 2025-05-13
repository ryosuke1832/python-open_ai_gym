import datetime
import os
import sys

import numpy as np
import tensorflow as tf

# /workspaces/Python_openAi_gym/main.py
from src.agents.ddpg_agent import DDPGAgent
from src.common.utils import cleanup_xvfb, setup_virtual_display
from src.environments.mountain_car import MountainCarContinuousEnv

# 現在のスクリプトの場所を取得してプロジェクトルートに移動
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)


# 再現性のためのシード設定
def set_seeds(seed=42):
    import random

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# メイン実行部分
if __name__ == "__main__":
    # タイムスタンプを生成（YYYY-MM-DD_HH-MM-SS形式）
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 仮想ディスプレイのセットアップ
    print("Xvfbクリーンアップを実行中...")
    cleanup_xvfb()
    setup_virtual_display()

    # 再現性のためのシード設定
    set_seeds(42)

    # Matplotlibの設定変更
    import matplotlib as mpl

    mpl.use("Agg")  # GUIバックエンドを使用しない設定
    mpl.rcParams["font.family"] = "DejaVu Sans"  # 英語フォントを使用
    mpl.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策

    # 環境の作成
    env = MountainCarContinuousEnv()

    # DDPGエージェントの作成（タイムスタンプを含む名前のディレクトリを指定）
    model_dir = f"saved_models/ddpg_mountaincar_{timestamp}"
    agent = DDPGAgent(env, model_dir=model_dir)

    # 以前のモデルをロードするかどうか
    load_pretrained = False  # 新しいモデルから開始
    if load_pretrained:
        # 特定のエピソードのモデルをロード
        episode_to_load = 100  # ロードしたいエピソード番号
        # ロード元のディレクトリを指定（タイムスタンプなしの元のディレクトリ）
        load_model_dir = "saved_models/ddpg_mountaincar"
        # 一時的にmodel_dirを変更
        original_model_dir = agent.model_dir
        agent.model_dir = load_model_dir
        agent.load_models(episode_to_load)
        agent.load_history()
        # model_dirを戻す
        agent.model_dir = original_model_dir
        start_episode = episode_to_load + 1
    else:
        start_episode = 0

    # 学習
    print(f"DDPG学習を開始... 結果は {model_dir} に保存されます")
    try:
        history = agent.train(
            episodes=650, max_steps=200, save_interval=10, start_episode=start_episode
        )
    except KeyboardInterrupt:
        print("学習が中断されました")
        # 中断時にもモデル保存
        agent.save_models(start_episode + len(agent.history["rewards"]) - 1)
        agent.save_history()

    # 評価
    print("\n学習したDDPGモデルの評価...")
    eval_rewards, eval_positions, eval_steps, success_rate = agent.evaluate(
        episodes=5, save_frames=True
    )

    # 全フレームからGIFアニメーションを作成
    try:
        import glob

        from PIL import Image

        frames_dir = f"{agent.model_dir}/frames"
        # タイムスタンプを含むアニメーション保存ディレクトリ
        animations_dir = f"animations_{timestamp}"
        os.makedirs(animations_dir, exist_ok=True)

        # 特定のエピソードのGIF作成
        for episode in [1, 20, 50, 100]:
            episode_dir = f"{frames_dir}/episode_{episode}"
            if os.path.exists(episode_dir):
                frames = sorted(
                    glob.glob(f"{episode_dir}/episode_{episode}_step_*.png")
                )
                if frames:
                    images = []
                    for frame in frames:
                        img = Image.open(frame)
                        images.append(img)

                    if images:
                        images[0].save(
                            f"{animations_dir}/ddpg_episode_{episode}.gif",
                            save_all=True,
                            append_images=images[1:],
                            optimize=False,
                            duration=100,
                            loop=0,
                        )
                        print(f"エピソード {episode} のGIFアニメーションを作成しました")

        # 成功エピソードのGIF作成
        success_dirs = glob.glob(f"{frames_dir}/success_*")
        for success_dir in success_dirs:
            episode = success_dir.split("_")[-1]
            frames = sorted(glob.glob(f"{success_dir}/episode_{episode}_step_*.png"))
            if frames:
                images = []
                for frame in frames:
                    img = Image.open(frame)
                    images.append(img)

                if images:
                    images[0].save(
                        f"{animations_dir}/ddpg_success_{episode}.gif",
                        save_all=True,
                        append_images=images[1:],
                        optimize=False,
                        duration=100,
                        loop=0,
                    )
                    print(f"成功エピソード {episode} のGIFアニメーションを作成しました")

    except Exception as e:
        print(f"GIF作成エラー: {e}")

    env.close()
