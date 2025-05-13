import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.agents.base_agent import BaseAgent
from src.common.buffer import ReplayBuffer
from src.common.utils import plot_training_progress, save_render_image


class DQNAgent(BaseAgent):
    """DQN Agent for discrete action spaces"""

    def __init__(self, env, model_dir="saved_models/dqn_cartpole"):
        # 環境の状態・行動空間の次元を取得
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # 親クラスの初期化
        super().__init__(env, state_dim, action_dim, model_dir)

        # 画像保存用のディレクトリ
        os.makedirs(f"{model_dir}/frames", exist_ok=True)

        # Q-networkとターゲットネットワークの構築
        self.q_network = self.build_q_network()
        self.target_q_network = self.build_q_network()

        # 初期ウェイトのコピー
        self.target_q_network.set_weights(self.q_network.get_weights())

        # オプティマイザ
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # リプレイバッファの初期化
        self.buffer = ReplayBuffer(
            buffer_capacity=10000, batch_size=64, state_dim=state_dim, action_dim=1
        )

        # ハイパーパラメータ
        self.gamma = 0.99  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率の減衰
        self.update_target_every = 10  # ターゲットネットワーク更新頻度
        self.train_step_counter = 0  # 学習ステップカウンタ

    def build_q_network(self):
        """Q-networkの構築"""
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(24, activation="relu")(inputs)
        x = layers.Dense(24, activation="relu")(x)
        outputs = layers.Dense(self.action_dim)(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")

        return model

    def policy(self, state, add_noise=True):
        """ε-グリーディ方策による行動選択"""
        if add_noise and np.random.rand() < self.epsilon:
            # ランダム行動（探索）
            return np.random.randint(0, self.action_dim)
        else:
            # 状態をバッチ形式に変換
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1))
            # Q値を計算し、最大のものを選択
            q_values = self.q_network(state_tensor)
            return np.argmax(q_values.numpy()[0])

    def remember(self, state, action, reward, next_state, done):
        """経験をバッファに追加"""
        self.buffer.record(state, [action], reward, next_state, done)

    def learn(self):
        """リプレイバッファからバッチを取り出して学習"""
        # バッファに十分なデータがない場合はスキップ
        if self.buffer.size() < self.buffer.batch_size:
            return

        # バッファからバッチを取得
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            self.buffer.sample()
        )

        # 現在のQ値を計算
        current_q = self.q_network(state_batch).numpy()

        # ターゲットQ値を計算
        target_q = np.copy(current_q)
        next_q = self.target_q_network(next_state_batch).numpy()
        max_next_q = np.max(next_q, axis=1)

        for i in range(self.buffer.batch_size):
            action = int(action_batch[i][0])
            if done_batch[i][0]:
                target_q[i][action] = reward_batch[i][0]
            else:
                target_q[i][action] = reward_batch[i][0] + self.gamma * max_next_q[i]

        # Q-networkを訓練
        self.q_network.fit(state_batch, target_q, epochs=1, verbose=0)

        # エクスプロレーション率を減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 一定間隔でターゲットネットワークを更新
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_every == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())

    def calculate_reward(self, state, next_state, raw_reward, done):
        """報酬の計算"""
        return self.env.calculate_reward(state, next_state, raw_reward, done)

    def save_models(self, episode):
        """モデルの保存"""
        self.q_network.save_weights(f"{self.model_dir}/q_network_episode_{episode}.h5")
        print(f"モデルを保存しました: エピソード {episode}")

    def load_models(self, episode):
        """モデルの読み込み"""
        try:
            q_path = f"{self.model_dir}/q_network_episode_{episode}.h5"
            if os.path.exists(q_path):
                self.q_network.load_weights(q_path)
                self.target_q_network.load_weights(q_path)
                print(f"モデルを読み込みました: エピソード {episode}")
                return True
            else:
                print(f"モデルが見つかりません: エピソード {episode}")
                return False
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False

    def train(self, episodes=1000, max_steps=500, save_interval=100, start_episode=0):
        """エージェントの学習"""
        # 学習履歴の初期化・復元
        if start_episode == 0:
            self.history = {
                "rewards": [],
                "avg_rewards": [],
                "steps": [],
            }

        # 特定のエピソードでフレームを保存するためのリスト
        save_frames_episodes = [1, 5, 10, 50, 100, 200]

        for episode in range(start_episode, episodes):
            # 環境のリセット
            observation = self.env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]

            episode_reward = 0
            episode_steps = 0

            # このエピソードの画像を保存するかどうか
            save_frames = episode in save_frames_episodes or episode == episodes - 1

            for step in range(max_steps):
                # 行動の選択
                action = self.policy(observation, add_noise=True)

                # 前の状態を保存
                prev_observation = observation

                # 行動の実行
                next_observation, raw_reward, done, info = self.env.step(action)
                if isinstance(info, tuple):
                    info = info[0]

                # カスタム報酬の計算
                reward = self.calculate_reward(
                    observation, next_observation, raw_reward, done
                )

                # 画像保存
                if save_frames and step % 5 == 0:
                    save_render_image(
                        self.env,
                        episode,
                        step,
                        next_observation,
                        reward,
                        action,
                        done,
                        directory=f"{self.model_dir}/frames/episode_{episode}",
                    )

                # 経験の保存
                self.remember(observation, action, reward, next_observation, done)

                # 学習
                self.learn()

                # 状態の更新
                observation = next_observation
                episode_reward += raw_reward
                episode_steps += 1

                if done:
                    break

            # エピソード終了後の処理
            self.history["rewards"].append(episode_reward)
            self.history["steps"].append(episode_steps)

            # 平均報酬の計算
            if len(self.history["rewards"]) >= 100:
                avg_reward = np.mean(self.history["rewards"][-100:])
            else:
                avg_reward = np.mean(self.history["rewards"])

            self.history["avg_rewards"].append(avg_reward)

            # 進捗報告
            if episode % 10 == 0 or episode == episodes - 1:
                print(
                    f"エピソード {episode}: 報酬 = {episode_reward:.2f}, 平均報酬 = {avg_reward:.2f}, ステップ数 = {episode_steps}"
                )

            # 定期的にモデルを保存
            if episode % save_interval == 0 or episode == episodes - 1:
                self.save_models(episode)

                # 学習データの保存
                self.save_history()

                # グラフの更新と保存
                if episode > 0:
                    plot_training_progress(
                        self.history,
                        save_path=f"{self.model_dir}/training_progress.png",
                        show=False,
                    )

            # 十分な報酬を得られるようになったら早期終了
            if avg_reward >= 475.0 and episode >= 100:
                print(
                    f"目標達成: エピソード {episode} で平均報酬 {avg_reward:.2f} を達成"
                )
                self.save_models(episode)
                self.save_history()
                break

        return self.history

    def evaluate(self, episodes=20, render=True, save_frames=True):
        """学習したエージェントの評価"""
        rewards = []
        steps_list = []

        # 評価用フレーム保存ディレクトリ
        eval_frames_dir = f"{self.model_dir}/frames/evaluation"
        if save_frames:
            os.makedirs(eval_frames_dir, exist_ok=True)

        for episode in range(episodes):
            observation = self.env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]

            episode_reward = 0

            for t in range(500):  # CartPoleの最大ステップ数
                if render:
                    self.env.render()
                    import time

                    time.sleep(0.01)  # 表示をゆっくりに

                # 行動を選択（ノイズなし）
                action = self.policy(observation, add_noise=False)

                # フレーム保存
                if save_frames and t % 5 == 0:
                    save_render_image(
                        self.env,
                        f"eval_{episode}",
                        t,
                        observation,
                        episode_reward,
                        action,
                        False,
                        directory=eval_frames_dir,
                    )

                # 行動を実行
                observation, reward, done, info = self.env.step(action)

                episode_reward += reward

                if done:
                    # 終了時のフレームを保存
                    if save_frames:
                        save_render_image(
                            self.env,
                            f"eval_{episode}",
                            t + 1,
                            observation,
                            episode_reward,
                            action,
                            True,
                            directory=eval_frames_dir,
                        )
                    break

            rewards.append(episode_reward)
            steps_list.append(t + 1)

            print(
                f"評価エピソード {episode + 1}: 報酬 = {episode_reward:.2f}, ステップ数 = {t + 1}"
            )

        print(
            f"評価結果: 平均報酬 = {np.mean(rewards):.2f}, 平均ステップ数 = {np.mean(steps_list):.1f}"
        )

        return rewards, steps_list
