import datetime
import glob
import os
import re
import sys

import numpy as np
import tensorflow as tf

from src.agents.ddpg_agent import DDPGAgent
from src.common.utils import (
    check_xvfb_dependencies,
    cleanup_xvfb,
    setup_virtual_display,
)
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


# 最新のモデルディレクトリを見つける関数
def find_latest_model_dir(base_dir="saved_models", prefix="ddpg_mountaincar"):
    dirs = glob.glob(f"{base_dir}/{prefix}_*")
    if not dirs:
        return None
    # 修正時刻が最新のディレクトリを選択
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir


# 全モデルディレクトリを見つける関数
def find_all_model_dirs(base_dir="saved_models", prefix="ddpg_mountaincar"):
    dirs = glob.glob(f"{base_dir}/{prefix}_*")
    # 修正時刻順にソート（最新が最後）
    return sorted(dirs, key=os.path.getmtime)


# ディレクトリ内のファイルを表示する関数
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
        f"{directory}/actor_episode_*.h5",  # DDPGのアクターモデル
        f"{directory}/critic_episode_*.h5",  # DDPGのクリティックモデル
        f"{directory}/*_episode_*.h5",  # より一般的なパターン
        f"{directory}/*.h5",  # すべての.h5ファイル
    ]

    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files

    return []


def natural_sort_key(s):
    """
    自然順（数値順）でソートするためのキー関数
    例: ['file1', 'file10', 'file2'] → ['file1', 'file2', 'file10']
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def extract_step_number(filepath):
    """ファイルパスからステップ番号を抽出する関数"""
    filename = os.path.basename(filepath)
    try:
        # 「episode_X_step_Y.png」からYを抽出
        step_str = filename.split("_step_")[1].split(".")[0]
        return int(step_str)
    except:
        # エラーの場合は0を返す
        return 0


# メイン実行部分
if __name__ == "__main__":
    # 仮想ディスプレイのセットアップ
    print("Xvfbクリーンアップを実行中...")
    cleanup_xvfb()
    setup_virtual_display()

    # # Xvfbの依存関係をチェック・インストール
    check_xvfb_dependencies()

    # 再現性のためのシード設定
    set_seeds(42)

    # Matplotlibの設定変更
    import matplotlib as mpl

    mpl.use("Agg")  # GUIバックエンドを使用しない設定
    mpl.rcParams["font.family"] = "DejaVu Sans"  # 英語フォントを使用
    mpl.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策

    # 環境の作成
    env = MountainCarContinuousEnv()

    # saved_modelsディレクトリの存在確認
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
        print("'saved_models'ディレクトリを作成しました")

    # 利用可能なモデルディレクトリを取得
    model_dirs = find_all_model_dirs()

    # タイムスタンプを生成（YYYY-MM-DD_HH-MM-SS形式）
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

                    # 一時的な新しいディレクトリを作成
                    temp_model_dir = f"saved_models/ddpg_mountaincar_{timestamp}"
                    agent = DDPGAgent(env, model_dir=temp_model_dir)

                    try:
                        # 一時的にmodel_dirを変更して、既存モデルを読み込む
                        original_model_dir = agent.model_dir
                        agent.model_dir = model_dir
                        load_success = agent.load_models(episode_to_load)
                        history_success = agent.load_history()
                        # model_dirを元に戻す
                        agent.model_dir = original_model_dir

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
                    # 新しいディレクトリを作成
                    model_dir = f"saved_models/ddpg_mountaincar_{timestamp}"
                    agent = DDPGAgent(env, model_dir=model_dir)
            else:
                print(
                    f"ディレクトリ {model_dir} にモデルファイルが見つかりませんでした。新規学習を開始します。"
                )
                # 新しいディレクトリを作成
                model_dir = f"saved_models/ddpg_mountaincar_{timestamp}"
                agent = DDPGAgent(env, model_dir=model_dir)
        else:
            # 新規ディレクトリを作成
            print("新規学習を開始します。")
            model_dir = f"saved_models/ddpg_mountaincar_{timestamp}"
            agent = DDPGAgent(env, model_dir=model_dir)
    else:
        print("既存のモデルディレクトリが見つかりませんでした。新規学習を開始します。")
        # 新しいディレクトリを作成
        model_dir = f"saved_models/ddpg_mountaincar_{timestamp}"
        agent = DDPGAgent(env, model_dir=model_dir)

    # 学習
    print(
        f"DDPG学習を開始... 結果は {model_dir} に保存されます (エピソード {start_episode} から)"
    )
    try:
        history = agent.train(
            episodes=650, max_steps=200, save_interval=10, start_episode=start_episode
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
            print("仮想ディスプレイが利用できないため、レンダリングなしで評価します")
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

        # アニメーション保存ディレクトリ
        animations_dir = "animations"
        os.makedirs(animations_dir, exist_ok=True)

        frames_dir = f"{agent.model_dir}/frames"
        if os.path.exists(frames_dir):
            # 特定のエピソードのGIF作成
            for episode in [1, 20, 50, 100, 200, 500]:
                episode_dir = f"{frames_dir}/episode_{episode}"
                if os.path.exists(episode_dir):
                    # ファイルのパターンを指定
                    frame_pattern = f"{episode_dir}/episode_{episode}_step_*.png"
                    frames = glob.glob(frame_pattern)

                    if frames:
                        # ステップ番号でソート
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
                                f"ddpg_mountaincar_{dir_name}_episode_{episode}.gif"
                            )

                            # デバッグ情報を表示
                            print(f"エピソード {episode} のフレーム順序:")
                            for i, frame in enumerate(frames[:5]):
                                print(f"  {i + 1}. {os.path.basename(frame)}")
                            if len(frames) > 5:
                                print(f"  ... 他 {len(frames) - 5} フレーム")

                            # GIF作成 - 再生速度を調整（持続時間を200msに）
                            images[0].save(
                                f"{animations_dir}/{gif_filename}",
                                save_all=True,
                                append_images=images[1:],
                                optimize=False,
                                duration=200,  # 再生速度を調整
                                loop=0,
                            )
                            print(
                                f"エピソード {episode} のGIFアニメーションを作成しました: {gif_filename}"
                            )

            # 成功エピソードのGIF作成
            success_dirs = glob.glob(f"{frames_dir}/success_*")
            for success_dir in success_dirs:
                episode = success_dir.split("_")[-1]
                # ファイルのパターンを指定
                frames = glob.glob(f"{success_dir}/episode_{episode}_step_*.png")
                if frames:
                    # ステップ番号でソート
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
                            f"ddpg_mountaincar_{dir_name}_success_{episode}.gif"
                        )

                        # GIF作成 - 再生速度を調整（持続時間を200msに）
                        images[0].save(
                            f"{animations_dir}/{gif_filename}",
                            save_all=True,
                            append_images=images[1:],
                            optimize=False,
                            duration=200,  # 再生速度を調整
                            loop=0,
                        )
                        print(
                            f"成功エピソード {episode} のGIFアニメーションを作成しました: {gif_filename}"
                        )

            # 評価エピソードのGIF作成（新規追加）
            eval_dir = f"{frames_dir}/evaluation"
            if os.path.exists(eval_dir):
                # 評価エピソードのパターンを検出
                eval_patterns = glob.glob(f"{eval_dir}/eval_*_step_*.png")
                eval_episodes = set()
                for path in eval_patterns:
                    filename = os.path.basename(path)
                    match = re.search(r"eval_(\d+)_", filename)
                    if match:
                        eval_episodes.add(match.group(1))

                # 各評価エピソードについてGIF作成
                for episode in eval_episodes:
                    # ファイルのパターンを指定
                    frames = glob.glob(f"{eval_dir}/eval_{episode}_step_*.png")
                    if frames:
                        # ステップ番号でソート
                        frames = sorted(frames, key=extract_step_number)

                        # 画像の読み込み
                        images = []
                        for frame in frames:
                            img = Image.open(frame)
                            images.append(img)

                        if images:
                            # モデルディレクトリ名を取得
                            dir_name = os.path.basename(model_dir)
                            # GIFファイル名
                            gif_filename = (
                                f"ddpg_mountaincar_{dir_name}_eval_{episode}.gif"
                            )

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
                                f"評価エピソード {episode} のGIFアニメーションを作成しました: {gif_filename}"
                            )
        else:
            print(f"フレームディレクトリが見つかりません: {frames_dir}")

    except Exception as e:
        print(f"GIF作成エラー: {e}")

    env.close()
    print("プログラムが正常に終了しました")
