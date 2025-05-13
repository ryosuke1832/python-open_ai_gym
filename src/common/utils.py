import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_render_image(
    env, episode, step, observation, reward, action, done, directory="frames"
):
    """環境の状態を画像として保存する関数"""
    os.makedirs(directory, exist_ok=True)

    # actionが単一の値かどうかをチェック
    action_text = (
        f"Action: {action[0]:.4f}"
        if hasattr(action, "__len__")
        else f"Action: {action:.4f}"
    )
    text_info = f"Reward: {reward:.2f} {action_text} Done: {done}"

    # rgb_arrayモードでレンダリング
    rgb_array = env.render(mode="rgb_array")

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

    text_info = f"Reward: {reward:.2f} {action_text} Done: {done}"

    # テキストサイズの計算
    try:
        # 新しいバージョン
        text_width = draw.textlength(text, font=font)
        text_height = font.getsize(text)[1]
    except AttributeError:
        try:
            # 中間のバージョン
            text_width, text_height = draw.textsize(text, font=font)
        except:
            # フォールバック
            text_width, text_height = 300, 20

    # テキスト位置
    position1 = (10, 10)
    position2 = (10, 40)
    position3 = (10, 70)

    # 背景と描画
    for i, (pos, txt) in enumerate(
        [(position1, text), (position2, text_state), (position3, text_info)]
    ):
        draw.rectangle(
            [
                pos[0] - 5,
                pos[1] - 5,
                pos[0] + text_width + 100,
                pos[1] + text_height + 5,
            ],
            fill=(0, 0, 0, 128),
        )
        draw.text(pos, txt, font=font, fill=(255, 255, 255))

    # 画像として保存
    img.save(f"{directory}/episode_{episode}_step_{step}.png")


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

    # 仮想ディスプレイの設定
    os.environ["DISPLAY"] = ":1"
    os.system("Xvfb :1 -screen 0 1024x768x24 &")
