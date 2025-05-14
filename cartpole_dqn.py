import glob
import os
import re
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


# 全モデルディレクトリを見つける関数（新規追加）
def find_all_model_dirs(base_dir="saved_models", prefix="dqn_cartpole"):
    dirs = glob.glob(f"{base_dir}/{prefix}_*")
    # 修正時刻順にソート（最新が最後）
    return sorted(dirs, key=os.path.getmtime)


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


def check_xvfb_dependencies():
    """Xvfbの依存関係をチェック・インストール"""
    import subprocess

    dependencies = [
        "xvfb",
        "x11-utils",
        "python3-opengl",
        "libgl1-mesa-glx",
        "libgl1-mesa-dev",
        "xdpyinfo",
    ]

    missing_deps = []
    for dep in dependencies:
        try:
            result = subprocess.run(
                ["dpkg", "-s", dep], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                missing_deps.append(dep)
        except:
            missing_deps.append(dep)

    if missing_deps:
        print(f"以下の依存パッケージが見つかりません: {', '.join(missing_deps)}")
        try:
            print("必要なパッケージをインストールしています...")
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y"] + missing_deps, check=True)
            print("パッケージのインストールが完了しました")
            return True
        except Exception as e:
            print(f"パッケージのインストールに失敗しました: {e}")
            return False

    return True


def natural_sort_key(s):
    """
    自然順（数値順）でソートするためのキー関数
    例: ['file1', 'file10', 'file2'] → ['file1', 'file2', 'file10']
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


# メイン実行部分
if __name__ == "__main__":
    # 仮想ディスプレイのセットアップ
    print("Xvfbクリーンアップを実行中...")
    cleanup_xvfb()
    setup_virtual_display()

    # Xvfbの依存関係をチェック・インストール
    check_xvfb_dependencies()

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

    # 利用可能なモデルディレクトリを取得
    model_dirs = find_all_model_dirs()

    # 新規学習を開始するか、既存のモデルから続きを学習するかのユーザー選択
    start_episode = 0
    if model_dirs:
        print("\n利用可能なモデルディレクトリ:")
        for i, dir_path in enumerate(model_dirs):
            dir_name = os.path.basename(dir_path)
            print(f"{i + 1}. {dir_name}")

        print(f"{len(model_dirs) + 1}. 新規学習を開始")

        while True:
            try:
                choice = int(input("\n選択肢の番号を入力してください: "))
                if 1 <= choice <= len(model_dirs) + 1:
                    break
                else:
                    print(f"1から{len(model_dirs) + 1}の間の番号を入力してください。")
            except ValueError:
                print("有効な番号を入力してください。")

        if choice <= len(model_dirs):
            # 既存のモデルディレクトリを使用
            model_dir = model_dirs[choice - 1]
            print(f"選択されたモデルディレクトリ: {os.path.basename(model_dir)}")

            # ディレクトリの内容を表示
            print_directory_contents(model_dir)

            # モデルファイルを検索
            model_files = find_model_files(model_dir)

            if model_files:
                print("検出されたモデルファイル:")
                for file in model_files:
                    print(f" - {os.path.basename(file)}")

                # エピソード番号を抽出
                episode_nums = []
                for file in model_files:
                    try:
                        filename = os.path.basename(file)
                        if "_episode_" in filename:
                            episode_num = int(
                                filename.split("_episode_")[1].split(".")[0]
                            )
                            episode_nums.append(episode_num)
                    except Exception as e:
                        print(
                            f"ファイル '{file}' からエピソード番号を抽出できませんでした: {e}"
                        )
                        continue

                if episode_nums:
                    episode_to_load = max(episode_nums)
                    print(f"最新のエピソード番号: {episode_to_load}")

                    agent = DQNAgent(env, model_dir=model_dir)

                    try:
                        load_success = agent.load_models(episode_to_load)
                        history_success = agent.load_history()

                        if load_success and history_success:
                            print("モデルと履歴を正常にロードしました")
                            start_episode = episode_to_load + 1
                        else:
                            print(
                                "モデルまたは履歴のロードに失敗しました。続行しますか？"
                            )
                            continue_choice = (
                                input("続行する場合は 'y' を入力: ").strip().lower()
                            )
                            if continue_choice != "y":
                                print("プログラムを終了します")
                                sys.exit(0)
                    except Exception as e:
                        print(f"モデルロード中にエラーが発生しました: {e}")
                        print("プログラムを終了します")
                        sys.exit(1)
                else:
                    print(
                        "モデルファイルからエピソード番号を抽出できませんでした。新規学習を開始します。"
                    )
                    agent = DQNAgent(env, model_dir=model_dir)
            else:
                print(
                    f"ディレクトリ {model_dir} にモデルファイルが見つかりませんでした。新規学習を開始します。"
                )
                agent = DQNAgent(env, model_dir=model_dir)
        else:
            # 新規ディレクトリを作成
            print("新規学習を開始します。")
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_dir = f"saved_models/dqn_cartpole_{timestamp}"
            agent = DQNAgent(env, model_dir=model_dir)
    else:
        print("既存のモデルディレクトリが見つかりませんでした。新規学習を開始します。")
        # 新しいディレクトリを作成
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
        # 評価前に仮想ディスプレイを再設定
        print("評価前に仮想ディスプレイを再設定しています...")
        display_available = setup_virtual_display()

        # 評価
        print("\n学習したDDPGモデルの評価...")
        try:
            if display_available:
                # 通常評価（レンダリングあり）
                eval_rewards, eval_positions, eval_steps, success_rate = agent.evaluate(
                    episodes=5, save_frames=True
                )
            else:
                # レンダリングなしの評価
                print(
                    "仮想ディスプレイが利用できないため、レンダリングなしで評価します"
                )
                eval_rewards, eval_positions, eval_steps, success_rate = agent.evaluate(
                    episodes=5, render=False, save_frames=False
                )
        except Exception as e:
            print(f"評価中にエラーが発生しました: {e}")
            print("レンダリングなしでの評価を試みます...")
            try:
                eval_rewards, eval_positions, eval_steps, success_rate = agent.evaluate(
                    episodes=5, render=False, save_frames=False
                )
            except Exception as e2:
                print(f"レンダリングなしの評価も失敗しました: {e2}")

    # 全フレームからGIFアニメーションを作成
    try:
        import glob

        from PIL import Image

        # 既存のanimationsディレクトリを使用するか、なければ作成
        animations_dir = "animations"
        os.makedirs(animations_dir, exist_ok=True)

        # GIF作成
        frames_dir = f"{agent.model_dir}/frames"
        if os.path.exists(frames_dir):
            # 特定のエピソードのGIF作成
            for episode in [1, 10, 100, 250, 500]:
                episode_dir = f"{frames_dir}/episode_{episode}"
                if os.path.exists(episode_dir):
                    # ファイルのパターンを指定
                    frame_pattern = f"{episode_dir}/episode_{episode}_step_*.png"
                    frames = glob.glob(frame_pattern)

                    if frames:
                        # 数値順にソート - ここがポイント
                        def extract_step_number(filepath):
                            filename = os.path.basename(filepath)
                            # 「episode_X_step_Y.png」からYを抽出
                            step_str = filename.split("_step_")[1].split(".")[0]
                            return int(step_str)

                        # ステップ番号で数値的にソート
                        frames = sorted(frames, key=extract_step_number)

                        # 画像の読み込み
                        images = []
                        for frame in frames:
                            img = Image.open(frame)
                            images.append(img)

                        if images:
                            # モデルディレクトリ名を取得
                            dir_name = os.path.basename(model_dir)
                            # GIFファイル名にモデルディレクトリ名を含める
                            gif_filename = (
                                f"dqn_cartpole_{dir_name}_episode_{episode}.gif"
                            )

                            # デバッグ情報を表示（任意）
                            print(f"エピソード {episode} のフレーム順序:")
                            for i, frame in enumerate(
                                frames[:5]
                            ):  # 最初の5フレームだけ表示
                                print(f"  {i + 1}. {os.path.basename(frame)}")
                            if len(frames) > 5:
                                print(f"  ... 他 {len(frames) - 5} フレーム")

                            # GIF作成
                            images[0].save(
                                f"{animations_dir}/{gif_filename}",
                                save_all=True,
                                append_images=images[1:],
                                optimize=False,
                                duration=200,
                                loop=0,
                            )
                            print(
                                f"エピソード {episode} のGIFアニメーションを作成しました: {gif_filename}"
                            )
        else:
            print(f"フレームディレクトリが見つかりません: {frames_dir}")

    except Exception as e:
        print(f"GIF作成エラー: {e}")

    env.close()
    print("プログラムが正常に終了しました")
