import glob
import os
import sys

import numpy as np
import tensorflow as tf

from src.agents.dqn_agent import DQNAgent
from src.common.utils import cleanup_xvfb, setup_virtual_display
from src.environments.cartpole import CartPoleEnv

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


# 最新のモデルディレクトリを見つける関数
def find_latest_model_dir(base_dir="saved_models", prefix="dqn_cartpole"):
    dirs = glob.glob(f"{base_dir}/{prefix}_*")
    if not dirs:
        return None
    # 修正時刻が最新のディレクトリを選択
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir


# ディレクトリ内のファイルを表示する関数（デバッグ用）
def print_directory_contents(directory):
    print(f"\n===== {directory} の内容 =====")
    if os.path.exists(directory):
        files = os.listdir(directory)
        for file in files:
            print(f" - {file}")
    else:
        print(f"ディレクトリ {directory} は存在しません")
    print("============================\n")


# モデルファイルを検索する関数
def find_model_files(directory):
    # 複数のパターンを試す
    patterns = [
        f"{directory}/q_network_episode_*.h5",  # 通常のパターン
        f"{directory}/*_network_*.h5",  # より一般的なパターン
        f"{directory}/*.h5",  # すべての.h5ファイル
    ]

    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files

    return []


# メイン実行部分
if __name__ == "__main__":
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
    env = CartPoleEnv()

    # saved_modelsディレクトリの存在確認
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
        print("'saved_models'ディレクトリを作成しました")

    # 最新のモデルディレクトリを検索
    latest_model_dir = find_latest_model_dir()

    # 以前のモデルを検出して続きから実行
    start_episode = 0

    if latest_model_dir:
        print(f"最新のモデルディレクトリを検出: {latest_model_dir}")

        # ディレクトリの内容を表示（デバッグ用）
        print_directory_contents(latest_model_dir)

        # モデルファイルを検索
        model_files = find_model_files(latest_model_dir)

        if model_files:
            print("検出されたモデルファイル:")
            for file in model_files:
                print(f" - {os.path.basename(file)}")

            # ファイル名からエピソード番号を抽出して最大値を取得
            episode_nums = []
            for file in model_files:
                try:
                    # ファイル名からエピソード番号を抽出
                    filename = os.path.basename(file)
                    if "_episode_" in filename:
                        episode_num = int(filename.split("_episode_")[1].split(".")[0])
                        episode_nums.append(episode_num)
                except Exception as e:
                    print(
                        f"ファイル '{file}' からエピソード番号を抽出できませんでした: {e}"
                    )
                    continue

            if episode_nums:
                episode_to_load = max(episode_nums)
                print(f"最新のエピソード番号: {episode_to_load}")

                # 既存のディレクトリを使用して続きから実行
                model_dir = latest_model_dir
                agent = DQNAgent(env, model_dir=model_dir)

                try:
                    load_success = agent.load_models(episode_to_load)
                    history_success = agent.load_history()

                    if load_success and history_success:
                        print("モデルと履歴を正常にロードしました")
                        start_episode = episode_to_load + 1
                    else:
                        print(
                            "モデルまたは履歴のロードに失敗しました。新規学習を開始します。"
                        )
                except Exception as e:
                    print(f"モデルロード中にエラーが発生しました: {e}")
            else:
                print(
                    "モデルファイルからエピソード番号を抽出できませんでした。新規学習を開始します。"
                )
                # 既存のディレクトリを使用
                model_dir = latest_model_dir
                agent = DQNAgent(env, model_dir=model_dir)
        else:
            print(
                f"ディレクトリ {latest_model_dir} にモデルファイルが見つかりませんでした。新規学習を開始します。"
            )
            # 既存のディレクトリを使用
            model_dir = latest_model_dir
            agent = DQNAgent(env, model_dir=model_dir)
    else:
        print("以前のモデルディレクトリが見つかりませんでした。新規学習を開始します。")
        # 新しいディレクトリを作成（最初の実行時のみ）
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = f"saved_models/dqn_cartpole_{timestamp}"
        agent = DQNAgent(env, model_dir=model_dir)

    # 学習
    print(
        f"DQN学習を開始... 結果は {model_dir} に保存されます (エピソード {start_episode} から)"
    )
    try:
        history = agent.train(
            episodes=500, max_steps=500, save_interval=10, start_episode=start_episode
        )
    except KeyboardInterrupt:
        print("学習が中断されました")
        # 中断時にもモデル保存
        current_episode = start_episode + len(agent.history["rewards"]) - 1
        print(f"エピソード {current_episode} でモデルを保存します")
        agent.save_models(current_episode)
        agent.save_history()

    # 評価
    print("\n学習したDQNモデルの評価...")
    eval_rewards, eval_steps = agent.evaluate(episodes=10, save_frames=True)

    # 全フレームからGIFアニメーションを作成
    try:
        import glob

        from PIL import Image

        # 既存のanimationsディレクトリを使用するか、なければ作成
        animations_dir = "animations"
        os.makedirs(animations_dir, exist_ok=True)

        frames_dir = f"{agent.model_dir}/frames"

        # 特定のエピソードのGIF作成
        for episode in [1, 5, 10, 50, 100]:
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
                        # モデルディレクトリ名を取得
                        dir_name = os.path.basename(model_dir)
                        # GIFファイル名にモデルディレクトリ名を含める
                        gif_filename = f"dqn_cartpole_{dir_name}_episode_{episode}.gif"
                        images[0].save(
                            f"{animations_dir}/{gif_filename}",
                            save_all=True,
                            append_images=images[1:],
                            optimize=False,
                            duration=100,
                            loop=0,
                        )
                        print(
                            f"エピソード {episode} のGIFアニメーションを作成しました: {gif_filename}"
                        )

    except Exception as e:
        print(f"GIF作成エラー: {e}")

    env.close()
