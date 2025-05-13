import os
import random
import signal
import subprocess
import time

import gym
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import layers


# Xvfbプロセスをクリーンアップする関数
def cleanup_xvfb():
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


# プログラムの開始時にクリーンアップを実行
print("Xvfbクリーンアップを実行中...")
cleanup_xvfb()

# ここから通常のコードが続く
# 仮想ディスプレイの設定
os.environ["DISPLAY"] = ":1"
os.system("Xvfb :1 -screen 0 1024x768x24 &")


matplotlib.use("Agg")  # GUIバックエンドを使用しない設定


# Matplotlibの設定変更
mpl.rcParams["font.family"] = "DejaVu Sans"  # 英語フォントを使用
mpl.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策

# メモリ使用量を制限
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 再現性のためのシード設定
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


# 画像保存用の関数
def save_render_image(
    env, episode, step, observation, reward, action, done, directory="frames_ddpg"
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
    text_state = f"Position: {observation[0]:.4f} Velocity: {observation[1]:.4f}"
    text_info = f"Reward: {reward:.2f} Action: {action[0]:.4f} Done: {done}"

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

    # 背景を追加して読みやすくする（1行目）
    draw.rectangle(
        [
            position1[0] - 5,
            position1[1] - 5,
            position1[0] + text_width + 5,
            position1[1] + text_height + 5,
        ],
        fill=(0, 0, 0, 128),
    )

    # テキスト描画（1行目）
    draw.text(position1, text, font=font, fill=(255, 255, 255))

    # 2行目
    draw.rectangle(
        [
            position2[0] - 5,
            position2[1] - 5,
            position2[0] + text_width + 100,
            position2[1] + text_height + 5,
        ],
        fill=(0, 0, 0, 128),
    )
    draw.text(position2, text_state, font=font, fill=(255, 255, 255))

    # 3行目
    draw.rectangle(
        [
            position3[0] - 5,
            position3[1] - 5,
            position3[0] + text_width + 100,
            position3[1] + text_height + 5,
        ],
        fill=(0, 0, 0, 128),
    )
    draw.text(position3, text_info, font=font, fill=(255, 255, 255))

    # 画像として保存
    img.save(f"{directory}/episode_{episode}_step_{step}.png")


class OUActionNoise:
    """Ornstein-Uhlenbeckプロセスによるノイズ生成"""

    def __init__(self, mean, std_deviation, theta=0.15, dt=0.01, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class ReplayBuffer:
    """経験リプレイバッファ"""

    def __init__(self, buffer_capacity=10000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        # メモリの初期化
        self.buffer_counter = 0
        self.state_buffer = np.zeros(
            (self.buffer_capacity, 2)
        )  # MountainCarは2次元状態
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, 2))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    def record(self, state, action, reward, next_state, done):
        index = self.buffer_counter % self.buffer_capacity

        # 配列の形状を統一
        if isinstance(state, np.ndarray) and len(state.shape) > 1:
            state = state.flatten()
        if isinstance(action, np.ndarray) and len(action.shape) > 1:
            action = action.flatten()
        if isinstance(next_state, np.ndarray) and len(next_state.shape) > 1:
            next_state = next_state.flatten()

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = done
        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        return (
            self.state_buffer[batch_indices],
            self.action_buffer[batch_indices],
            self.reward_buffer[batch_indices],
            self.next_state_buffer[batch_indices],
            self.done_buffer[batch_indices],
        )

    def size(self):
        return min(self.buffer_counter, self.buffer_capacity)


class DDPGAgent:
    """DDPG Agent for MountainCar"""

    def __init__(self, env, model_dir="saved_models"):
        self.env = env
        self.state_dim = 2  # MountainCarは2次元状態
        self.action_dim = 1  # DDPGは連続値を出力
        self.action_bound = 1.0  # 行動の境界値
        self.model_dir = model_dir

        # ディレクトリがなければ作成
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 画像保存用のディレクトリ
        os.makedirs("frames_ddpg", exist_ok=True)

        # Actorとターゲットネットワークの構築
        self.actor_model = self.get_actor_model()
        self.target_actor = self.get_actor_model()

        # Criticとターゲットネットワークの構築
        self.critic_model = self.get_critic_model()
        self.target_critic = self.get_critic_model()

        # 初期ウェイトのコピー
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # オプティマイザの設定
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # リプレイバッファの初期化
        self.buffer = ReplayBuffer(50000, 128)

        # OUノイズプロセスの初期化
        # ノイズ設定の調整
        self.noise = OUActionNoise(
            mean=np.zeros(1), std_deviation=0.35 * np.ones(1)
        )  # 0.25から0.35に増加

        # ハイパーパラメーター
        self.gamma = 0.99  # 割引率
        self.tau = 0.005  # ターゲットネットワーク更新係数

        # 状態正規化用のパラメータ
        self.state_mean = np.array([-0.6, 0.0])  # 位置と速度の平均
        self.state_std = np.array([0.5, 0.035])  # 位置と速度の標準偏差

        # トレーニングデータ
        self.history = {
            "rewards": [],
            "avg_rewards": [],
            "positions": [],
            "steps": [],
            "successes": [],
        }

    # より大きなネットワークとより小さな学習率
    def get_actor_model(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation="relu")(inputs)  # 128から256に増加
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)  # 128から256に増加
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(self.action_dim, activation="tanh")(x)
        outputs = outputs * self.action_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic_model(self):
        """クリティックネットワークの定義（より軽量）"""
        # 状態の入力
        state_input = layers.Input(shape=(self.state_dim,))
        state_out = layers.Dense(32, activation="relu")(state_input)

        # 行動の入力
        action_input = layers.Input(shape=(self.action_dim,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # 状態と行動の結合
        concat = layers.Concatenate()([state_out, action_out])

        x = layers.Dense(128, activation="relu")(concat)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(1)(x)

        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def normalize_state(self, state):
        """状態の正規化"""
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def policy(self, state, add_noise=True, episode=None):
        """行動選択ポリシー"""
        # 状態の正規化
        norm_state = self.normalize_state(state.reshape(1, -1))
        sampled_actions = self.actor_model(norm_state)

        if add_noise:
            # 学習初期はより大きなノイズ
            noise_scale = 1.0
            if episode is not None and episode < 500:  # 300から500に延長
                noise_scale = 2.0 - episode / 250  # 徐々に減少（より緩やかに）

            noise = self.noise() * noise_scale
            sampled_actions = sampled_actions.numpy() + noise
        else:
            sampled_actions = sampled_actions.numpy()

        # クリップ
        sampled_actions = np.clip(
            sampled_actions, -self.action_bound, self.action_bound
        )

        # 形状の確認と修正 - 必ず1次元配列で返す
        return sampled_actions.flatten()  # 平坦化して返す

    def remember(self, state, action, reward, next_state, done):
        """経験をリプレイバッファに保存"""
        self.buffer.record(state, action, reward, next_state, done)

    def learn(self):
        """経験リプレイからネットワークを学習"""
        # バッファに十分なデータがない場合はスキップ
        if self.buffer.size() < self.buffer.batch_size:
            return

        # バッファからバッチをサンプリング
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            self.buffer.sample()
        )

        # 状態の正規化
        norm_state_batch = self.normalize_state(state_batch)
        norm_next_state_batch = self.normalize_state(next_state_batch)

        # TensorFlow GradientTapeを使用して勾配を監視
        with tf.GradientTape() as tape:
            # ターゲットアクションの計算
            target_actions = self.target_actor(norm_next_state_batch)
            # ターゲットQ値の計算
            target_critic_value = self.target_critic(
                [norm_next_state_batch, target_actions]
            )
            # Q値目標の計算
            target_q = reward_batch + self.gamma * target_critic_value * (
                1 - done_batch
            )
            # 現在のQ値の計算
            critic_value = self.critic_model([norm_state_batch, action_batch])
            # クリティックの損失計算
            critic_loss = tf.math.reduce_mean(tf.math.square(target_q - critic_value))

        # クリティックの勾配計算と適用
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        # アクターの学習
        with tf.GradientTape() as tape:
            # 現在の状態に対するアクションを予測
            actions = self.actor_model(norm_state_batch)
            # アクションの価値を計算
            critic_value = self.critic_model([norm_state_batch, actions])
            # アクターの損失（クリティックの出力の負値）
            actor_loss = -tf.math.reduce_mean(critic_value)

        # アクターの勾配計算と適用
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

        # ターゲットネットワークの更新
        self.update_target()

    def update_target(self):
        """ソフトターゲット更新"""
        # アクターのターゲット更新
        new_weights = []
        target_weights = self.target_actor.get_weights()
        for i, weight in enumerate(self.actor_model.get_weights()):
            new_weights.append(weight * self.tau + target_weights[i] * (1 - self.tau))
        self.target_actor.set_weights(new_weights)

        # クリティックのターゲット更新
        new_weights = []
        target_weights = self.target_critic.get_weights()
        for i, weight in enumerate(self.critic_model.get_weights()):
            new_weights.append(weight * self.tau + target_weights[i] * (1 - self.tau))
        self.target_critic.set_weights(new_weights)

    def calculate_custom_reward(self, state, next_state, raw_reward, done):
        position, velocity = next_state
        prev_position, prev_velocity = state

        # 基本報酬
        reward = raw_reward

        # 位置向上に対する報酬を強化
        if position > prev_position:
            reward += (position - prev_position) * 30  # 20から30に増加

        # 目標近くでの追加ボーナス（より段階的に）
        if position > 0.4:
            reward += 30.0  # 20から30に増加
        if position > 0.45:
            reward += 50.0  # より積極的に報酬を与える

        # 成功ボーナスをさらに増加
        if position >= 0.5:
            reward += 200.0  # 100から200に増加

        return reward

    def train(self, episodes=1000, max_steps=1000, save_interval=100, start_episode=0):
        """エージェントの学習"""
        # 学習履歴の初期化・復元
        if start_episode == 0:
            self.history = {
                "rewards": [],
                "avg_rewards": [],
                "positions": [],
                "steps": [],
                "successes": [],
            }

        # 特定のエピソードでフレームを保存するためのリスト
        save_frames_episodes = [1, 20, 40, 60, 80, 100]

        for episode in range(start_episode, episodes):
            # 環境のリセット
            observation = self.env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]

            total_reward = 0
            max_position = -np.inf
            success = False  # このエピソードが成功したかどうか

            # このエピソードの画像を保存するかどうか
            save_frames = episode in save_frames_episodes or episode == episodes - 1

            for step in range(max_steps):
                # 行動の選択
                continuous_action = self.policy(
                    observation, add_noise=True, episode=episode
                )

                # 前の状態を保存
                prev_observation = observation

                # 行動の実行
                next_observation, raw_reward, done, info = self.env.step(
                    continuous_action
                )
                if isinstance(info, tuple):
                    info = info[0]

                # カスタム報酬の計算
                reward = self.calculate_custom_reward(
                    observation, next_observation, raw_reward, done
                )

                # 画像保存（特定のエピソード、または成功時）
                if save_frames or (
                    success and step % 5 == 0
                ):  # 成功時は5ステップごとに保存
                    save_render_image(
                        self.env,
                        episode,
                        step,
                        next_observation,
                        reward,
                        continuous_action,
                        done,
                        directory=f"frames_ddpg/episode_{episode}",
                    )

                # 経験の保存
                self.remember(
                    observation, continuous_action, reward, next_observation, done
                )

                # ネットワークの学習
                self.learn()

                # 状態の更新
                observation = next_observation
                total_reward += raw_reward
                max_position = max(max_position, observation[0])

                # 成功判定 - 旗（0.5以上）に到達したら成功
                if observation[0] >= 0.5:
                    success = True
                    if not done:  # もし環境が終了していなければ
                        done = True
                        reward += 10.0  # 追加報酬

                    # 成功時は最後のフレームを必ず保存
                    save_render_image(
                        self.env,
                        episode,
                        step,
                        observation,
                        reward,
                        continuous_action,
                        done,
                        directory=f"frames_ddpg/success_{episode}",
                    )

                if done:
                    break

            # エピソード終了後の処理
            self.history["rewards"].append(total_reward)
            self.history["positions"].append(max_position)
            self.history["steps"].append(step + 1)
            self.history["successes"].append(1 if success else 0)

            # 平均報酬と成功率の計算
            if len(self.history["rewards"]) >= 100:
                avg_reward = np.mean(self.history["rewards"][-100:])
                success_rate = np.mean(self.history["successes"][-100:]) * 100
            else:
                avg_reward = np.mean(self.history["rewards"])
                success_rate = np.mean(self.history["successes"]) * 100

            self.history["avg_rewards"].append(avg_reward)

            # 進捗報告
            if episode % 10 == 0 or episode == episodes - 1:
                print(
                    f"エピソード {episode}: 報酬 = {total_reward:.2f}, 平均報酬 = {avg_reward:.2f}, 最大位置 = {max_position:.4f}, ステップ数 = {step + 1}, 成功率 = {success_rate:.1f}%"
                )

            # 定期的にモデルを保存
            if (
                episode % save_interval == 0
                or episode == episodes - 1
                or success_rate >= 30
            ):
                self.save_models(episode)

                # 学習データの保存
                self.save_history()

                # グラフの更新と保存
                if episode > 0:
                    self.plot_training_progress(show=False)

            # 一定の成功率に達したら早期終了
            if success_rate >= 30 and episode >= 50:  # より早く終了するために緩和
                print(
                    f"目標達成: エピソード {episode} で成功率 {success_rate:.1f}% を達成"
                )
                self.save_models(episode)
                self.save_history()

                # 成功時の画像も保存
                evaluation_frames_dir = "frames_ddpg/final_success"
                os.makedirs(evaluation_frames_dir, exist_ok=True)

                # 成功したモデルの実行を可視化
                observation = self.env.reset()
                if isinstance(observation, tuple):
                    observation = observation[0]

                for step in range(200):
                    continuous_action = self.policy(observation, add_noise=False)
                    discrete_action = self.continuous_to_discrete(continuous_action[0])

                    save_render_image(
                        self.env,
                        "final",
                        step,
                        observation,
                        0,
                        discrete_action,
                        False,
                        directory=evaluation_frames_dir,
                    )

                    next_observation, reward, done, _ = self.env.step(discrete_action)
                    observation = next_observation

                    if done or observation[0] >= 0.5:
                        # 最後のフレームを保存
                        save_render_image(
                            self.env,
                            "final",
                            step + 1,
                            observation,
                            reward,
                            discrete_action,
                            True,
                            directory=evaluation_frames_dir,
                        )
                        break

                break

        return self.history

    # continuous_to_discrete メソッドを追加する場合
    def continuous_to_discrete(self, continuous_action):
        """連続行動を離散行動に変換（評価用）"""
        return continuous_action  # MountainCarContinuous環境では変換不要

    def save_models(self, episode):
        """モデルの保存"""
        self.actor_model.save_weights(f"{self.model_dir}/actor_episode_{episode}.h5")
        self.critic_model.save_weights(f"{self.model_dir}/critic_episode_{episode}.h5")
        print(f"モデルを保存しました: エピソード {episode}")

    def load_models(self, episode):
        """モデルの読み込み"""
        try:
            actor_path = f"{self.model_dir}/actor_episode_{episode}.h5"
            critic_path = f"{self.model_dir}/critic_episode_{episode}.h5"

            if os.path.exists(actor_path) and os.path.exists(critic_path):
                self.actor_model.load_weights(actor_path)
                self.critic_model.load_weights(critic_path)
                self.target_actor.load_weights(actor_path)
                self.target_critic.load_weights(critic_path)
                print(f"モデルを読み込みました: エピソード {episode}")
                return True
            else:
                print(f"モデルが見つかりません: エピソード {episode}")
                return False
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False

    def save_history(self):
        """学習履歴の保存"""
        np.savez(
            f"{self.model_dir}/training_history.npz",
            rewards=np.array(self.history["rewards"]),
            avg_rewards=np.array(self.history["avg_rewards"]),
            positions=np.array(self.history["positions"]),
            steps=np.array(self.history["steps"]),
            successes=np.array(self.history["successes"]),
        )

    def load_history(self):
        """学習履歴の読み込み"""
        try:
            history_path = f"{self.model_dir}/training_history.npz"
            if os.path.exists(history_path):
                data = np.load(history_path)
                self.history["rewards"] = data["rewards"].tolist()
                self.history["avg_rewards"] = data["avg_rewards"].tolist()
                self.history["positions"] = data["positions"].tolist()
                self.history["steps"] = data["steps"].tolist()
                self.history["successes"] = data["successes"].tolist()
                print("学習履歴を読み込みました")
                return True
            else:
                print("学習履歴が見つかりません")
                return False
        except Exception as e:
            print(f"履歴読み込みエラー: {e}")
            return False

    def plot_training_progress(self, show=True):
        """学習の進捗をグラフ化（英語表記）"""
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(self.history["rewards"])
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")

        plt.subplot(2, 2, 2)
        plt.plot(self.history["avg_rewards"])
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Average Reward (100 episodes)")

        plt.subplot(2, 2, 3)
        plt.plot(self.history["positions"])
        plt.xlabel("Episode")
        plt.ylabel("Max Position")
        plt.title("Max Position per Episode")
        plt.axhline(y=0.5, color="r", linestyle="-", label="Goal Position")
        plt.legend()

        plt.subplot(2, 2, 4)
        success_rates = []
        for i in range(len(self.history["successes"])):
            if i < 100:
                success_rates.append(np.mean(self.history["successes"][: i + 1]) * 100)
            else:
                success_rates.append(
                    np.mean(self.history["successes"][i - 99 : i + 1]) * 100
                )
        plt.plot(success_rates)
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (%)")
        plt.title("Success Rate (100 episodes)")

        plt.tight_layout()
        plt.savefig(f"{self.model_dir}/training_progress.png")

        if show:
            plt.show()
        else:
            plt.close()

    def evaluate(self, episodes=20, render=True, save_frames=True):
        """学習したエージェントの評価"""
        rewards = []
        positions = []
        steps_list = []
        success_count = 0

        # 評価用フレーム保存ディレクトリ
        eval_frames_dir = "frames_ddpg/evaluation"
        if save_frames:
            os.makedirs(eval_frames_dir, exist_ok=True)

        for episode in range(episodes):
            observation = self.env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]

            total_reward = 0
            max_position = -np.inf
            success = False

            for t in range(1000):  # 長めに実行
                if render:
                    self.env.render()
                    time.sleep(0.01)  # 表示をゆっくりに
                # 行動を選択（ノイズなし）
                continuous_action = self.policy(observation, add_noise=False)

                # continuous_actionを常に適切な形式に変換
                if not isinstance(continuous_action, np.ndarray):
                    continuous_action = np.array([continuous_action])
                elif len(continuous_action.shape) == 0:
                    continuous_action = np.array([continuous_action.item()])

                # フレーム保存（continuous_actionを配列として保持）
                if save_frames and t % 5 == 0:
                    save_render_image(
                        self.env,
                        f"eval_{episode}",
                        t,
                        observation,
                        total_reward,
                        continuous_action,
                        False,
                        directory=eval_frames_dir,
                    )

                # 行動を実行 - 常に適切な形式
                observation, reward, done, info = self.env.step(continuous_action)

                total_reward += reward
                max_position = max(max_position, observation[0])

                # 成功判定
                if observation[0] >= 0.5:
                    success = True

                    # 成功時は最後のフレームを保存
                    if save_frames:
                        save_render_image(
                            self.env,
                            f"eval_{episode}_success",
                            t,
                            observation,
                            reward,
                            continuous_action,
                            True,
                            directory=eval_frames_dir,
                        )

                if done:
                    break

            rewards.append(total_reward)
            positions.append(max_position)
            steps_list.append(t + 1)

            if success:
                success_count += 1

            print(
                f"評価エピソード {episode + 1}: 報酬 = {total_reward:.2f}, 最大位置 = {max_position:.4f}, ステップ数 = {t + 1}, 成功 = {success}"
            )

        success_rate = success_count / episodes * 100
        print(
            f"評価結果: 平均報酬 = {np.mean(rewards):.2f}, 平均最大位置 = {np.mean(positions):.4f}, 平均ステップ数 = {np.mean(steps_list):.1f}, 成功率 = {success_rate:.1f}%"
        )

        # 評価結果をGIFに変換
        if save_frames:
            try:
                import glob

                from PIL import Image

                # 成功エピソードのフレームからGIFを作成
                success_frames = sorted(
                    glob.glob(f"{eval_frames_dir}/eval_*_success_*.png")
                )
                if success_frames:
                    images = []
                    for frame in success_frames:
                        img = Image.open(frame)
                        images.append(img)

                    # GIFとして保存
                    os.makedirs("animations", exist_ok=True)
                    if images:
                        images[0].save(
                            "animations/ddpg_success.gif",
                            save_all=True,
                            append_images=images[1:],
                            optimize=False,
                            duration=100,
                            loop=0,
                        )
                        print(
                            "成功エピソードのGIFアニメーションを作成しました: animations/ddpg_success.gif"
                        )
            except Exception as e:
                print(f"GIF作成エラー: {e}")

        return rewards, positions, steps_list, success_rate


# メイン実行部分
if __name__ == "__main__":
    # 環境の作成
    env = gym.make("MountainCarContinuous-v0")

    # DDPGエージェントの作成
    agent = DDPGAgent(env)

    # 以前のモデルをロードするかどうか
    load_pretrained = False  # 新しいモデルから開始
    if load_pretrained:
        # 特定のエピソードのモデルをロード
        episode_to_load = 100  # ロードしたいエピソード番号
        agent.load_models(episode_to_load)
        agent.load_history()
        start_episode = episode_to_load + 1
    else:
        start_episode = 0

    # 学習
    print("DDPG学習を開始...")
    try:
        history = agent.train(
            episodes=650, max_steps=200, save_interval=10, start_episode=start_episode
        )
    except KeyboardInterrupt:
        print("学習が中断されました")
        # 中断時にもモデル保存
        agent.save_models(start_episode + len(agent.history["rewards"]) - 1)
        agent.save_history()

    # 学習結果のグラフ表示
    agent.plot_training_progress(show=False)  # show=Falseでファイルのみ保存

    # 評価
    print("\n学習したDDPGモデルの評価...")
    eval_rewards, eval_positions, eval_steps, success_rate = agent.evaluate(
        episodes=5, save_frames=True
    )
    # 評価結果のグラフ表示部分を英語表記に修正
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.bar(range(1, len(eval_rewards) + 1), eval_rewards)
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Evaluation Episode")

    plt.subplot(1, 3, 2)
    plt.bar(range(1, len(eval_positions) + 1), eval_positions)
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Max Position")
    plt.title("Max Position per Evaluation Episode")
    plt.axhline(y=0.5, color="r", linestyle="-", label="Goal Position")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.bar(range(1, len(eval_steps) + 1), eval_steps)
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Number of Steps")
    plt.title("Steps per Evaluation Episode")

    plt.tight_layout()
    plt.savefig("mountaincar_ddpg_evaluation.png")
    plt.close()  # plt.show()の代わりにclose()を使用

    # 全フレームからGIFアニメーションを作成
    try:
        import glob

        from PIL import Image

        # 特定のエピソードのGIF作成
        for episode in [1, 20, 50, 100]:
            episode_dir = f"frames_ddpg/episode_{episode}"
            if os.path.exists(episode_dir):
                frames = sorted(
                    glob.glob(f"{episode_dir}/episode_{episode}_step_*.png")
                )
                if frames:
                    images = []
                    for frame in frames:
                        img = Image.open(frame)
                        images.append(img)

                    # GIFとして保存
                    os.makedirs("animations", exist_ok=True)
                    if images:
                        images[0].save(
                            f"animations/ddpg_episode_{episode}.gif",
                            save_all=True,
                            append_images=images[1:],
                            optimize=False,
                            duration=100,
                            loop=0,
                        )
                        print(f"エピソード {episode} のGIFアニメーションを作成しました")

        # 成功エピソードのGIF作成
        success_dirs = glob.glob("frames_ddpg/success_*")
        for success_dir in success_dirs:
            episode = success_dir.split("_")[-1]
            frames = sorted(glob.glob(f"{success_dir}/episode_{episode}_step_*.png"))
            if frames:
                images = []
                for frame in frames:
                    img = Image.open(frame)
                    images.append(img)

                # GIFとして保存
                os.makedirs("animations", exist_ok=True)
                if images:
                    images[0].save(
                        f"animations/ddpg_success_{episode}.gif",
                        save_all=True,
                        append_images=images[1:],
                        optimize=False,
                        duration=100,
                        loop=0,
                    )
                    print(f"成功エピソード {episode} のGIFアニメーションを作成しました")

        # 最終成功エピソードのGIF作成
        final_success_dir = "frames_ddpg/final_success"
        if os.path.exists(final_success_dir):
            frames = sorted(glob.glob(f"{final_success_dir}/episode_final_step_*.png"))
            if frames:
                images = []
                for frame in frames:
                    img = Image.open(frame)
                    images.append(img)

                # GIFとして保存
                if images:
                    images[0].save(
                        "animations/ddpg_final_success.gif",
                        save_all=True,
                        append_images=images[1:],
                        optimize=False,
                        duration=100,
                        loop=0,
                    )
                    print("最終成功エピソードのGIFアニメーションを作成しました")

    except Exception as e:
        print(f"GIF作成エラー: {e}")

    env.close()
