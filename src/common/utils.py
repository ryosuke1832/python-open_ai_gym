import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def check_xvfb_dependencies():
    """Xvfbの依存関係をチェック（コマンドの存在を確認）"""
    import shutil

    # 実行ファイルの存在をチェック（パッケージ名ではなく）
    dependencies = ["Xvfb", "xdpyinfo"]

    missing_deps = []
    for dep in dependencies:
        if shutil.which(dep) is None:
            missing_deps.append(dep)

    if missing_deps:
        print(f"以下のコマンドが見つかりません: {', '.join(missing_deps)}")
        return False
    else:
        print("必要なすべてのコマンドが利用可能です")
        return True


def save_render_image(
    env, episode, step, observation, reward, action, done, directory="frames"
):
    """環境の状態を画像として保存する関数"""
    try:
        os.makedirs(directory, exist_ok=True)

        # actionが単一の値かどうかをチェック
        action_text = (
            f"Action: {action[0]:.4f}"
            if hasattr(action, "__len__")
            else f"Action: {action:.4f}"
        )
        text_info = f"Reward: {reward:.2f} {action_text} Done: {done}"

        # rgb_arrayモードでレンダリング
        try:
            rgb_array = env.render(mode="rgb_array")
        except Exception as e:
            print(f"レンダリングエラー: {e}、ダミー画像を生成します")
            # エラー時はダミー画像を作成
            rgb_array = np.zeros((400, 600, 3), dtype=np.uint8)
            # 情報テキストを表示するためのスペース
            rgb_array[0:100, :] = [50, 50, 50]  # 上部に暗いグレーの帯

        # PIL画像に変換
        img = Image.fromarray(rgb_array)

        # 描画オブジェクトを作成
        draw = ImageDraw.Draw(img)

        # デフォルトフォントを使用
        try:
            # システムフォントがある場合
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except IOError:
            # デフォルトフォントを使用
            font = ImageFont.load_default()

        # テキストを描画（左上に配置）
        text = f"Episode: {episode} Step: {step}"

        # 状態テキストは汎用的に処理
        text_state = ""
        if isinstance(observation, np.ndarray):
            if observation.size <= 4:  # 小さい状態空間なら全部表示
                text_state = "State: " + " ".join([f"{x:.4f}" for x in observation])
            else:
                text_state = f"State shape: {observation.shape}"
        else:
            text_state = f"State: {observation}"

        # テキスト位置
        position1 = (10, 10)
        position2 = (10, 40)
        position3 = (10, 70)

        # 背景と描画（簡易化）
        draw.rectangle([0, 0, 600, 100], fill=(0, 0, 0, 128))
        draw.text(position1, text, font=font, fill=(255, 255, 255))
        draw.text(position2, text_state, font=font, fill=(255, 255, 255))
        draw.text(position3, text_info, font=font, fill=(255, 255, 255))

        # 画像として保存
        img.save(f"{directory}/episode_{episode}_step_{step}.png")
        return True
    except Exception as e:
        print(f"画像保存エラー: {e}")
        return False


def plot_training_progress(history, save_path="training_progress.png", show=True):
    """学習の進捗をグラフ化（英語表記）"""
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history["rewards"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")

    plt.subplot(2, 2, 2)
    plt.plot(history["avg_rewards"])
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward (100 episodes)")

    plt.subplot(2, 2, 3)
    plt.plot(history["positions"])
    plt.xlabel("Episode")
    plt.ylabel("Max Position")
    plt.title("Max Position per Episode")
    plt.axhline(y=0.5, color="r", linestyle="-", label="Goal Position")
    plt.legend()

    plt.subplot(2, 2, 4)
    success_rates = []
    for i in range(len(history["successes"])):
        if i < 100:
            success_rates.append(np.mean(history["successes"][: i + 1]) * 100)
        else:
            success_rates.append(np.mean(history["successes"][i - 99 : i + 1]) * 100)
    plt.plot(success_rates)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate (100 episodes)")

    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def cleanup_xvfb():
    """Xvfbプロセスをクリーンアップする関数"""
    import os
    import signal
    import subprocess

    try:
        # 既存のロックファイルを削除
        if os.path.exists("/tmp/.X1-lock"):
            os.remove("/tmp/.X1-lock")
            print("Removed /tmp/.X1-lock")

        if os.path.exists("/tmp/.X11-unix/X1"):
            os.remove("/tmp/.X11-unix/X1")
            print("Removed /tmp/.X11-unix/X1")

        # 実行中のXvfbプロセスを終了
        try:
            # プロセスIDを取得
            cmd = "pgrep Xvfb"
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            if output:
                for pid in output.decode().strip().split("\n"):
                    if pid:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"Terminated Xvfb process with PID: {pid}")
        except Exception as e:
            print(f"プロセス終了エラー: {e}")

    except Exception as e:
        print(f"Xvfb cleanup error: {e}")


def setup_virtual_display():
    """仮想ディスプレイをセットアップ"""
    import os
    import subprocess
    import time

    print("仮想ディスプレイのセットアップを開始...")

    # 既存のXvfbプロセスを確実に終了
    cleanup_xvfb()

    # 新しいディスプレイ番号を使用
    display_num = 1

    # ディスプレイ番号が使用可能になるまで試行
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            # Xvfbプロセスを起動
            xvfb_cmd = f"Xvfb :{display_num} -screen 0 1024x768x24"
            subprocess.Popen(
                xvfb_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # プロセスが起動するまで待機
            time.sleep(1)

            # ディスプレイが利用可能か確認
            check_cmd = f"xdpyinfo -display :{display_num}"
            check_process = subprocess.Popen(
                check_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            _, stderr = check_process.communicate()

            if check_process.returncode == 0:
                # ディスプレイが利用可能
                os.environ["DISPLAY"] = f":{display_num}"
                print(f"仮想ディスプレイが設定されました - DISPLAY=:{display_num}")
                return True

            # 次のディスプレイ番号を試す
            display_num += 1

        except Exception as e:
            print(f"ディスプレイ :{display_num} の設定中にエラーが発生: {e}")
            display_num += 1

    # 仮想ディスプレイの設定に失敗した場合
    print("警告: 仮想ディスプレイの設定に失敗しました。レンダリングなしで続行します。")
    os.environ["DISPLAY"] = ""
    return False
