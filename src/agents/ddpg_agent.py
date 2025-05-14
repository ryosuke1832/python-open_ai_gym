import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.agents.base_agent import BaseAgent
from src.common.buffer import ReplayBuffer
from src.common.noise import OUActionNoise
from src.common.utils import plot_training_progress, save_render_image


class DDPGAgent(BaseAgent):
    """DDPG Agent for continuous action spaces"""

    def __init__(self, env, model_dir="saved_models"):
        # 環境の状態・行動空間の次元を取得
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]

        # 親クラスの初期化
        super().__init__(env, state_dim, action_dim, model_dir)

        self.action_bound = action_bound

        # 画像保存用のディレクトリ
        os.makedirs(f"{model_dir}/frames", exist_ok=True)

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
        self.buffer = ReplayBuffer(50000, 128, state_dim, action_dim)

        # OUノイズプロセスの初期化
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim), std_deviation=0.35 * np.ones(action_dim)
        )

        # ハイパーパラメーター
        self.gamma = 0.99  # 割引率
        self.tau = 0.005  # ターゲットネットワーク更新係数

        # 状態正規化用のパラメータ (環境に依存する部分なので、環境別クラスへの移動を検討)
        self.state_mean = np.array([-0.6, 0.0])  # 位置と速度の平均
        self.state_std = np.array([0.5, 0.035])  # 位置と速度の標準偏差

    def get_actor_model(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(self.action_dim, activation="tanh")(x)
        outputs = outputs * self.action_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic_model(self):
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
            if episode is not None and episode < 500:
                noise_scale = 2.0 - episode / 250

            noise = self.noise() * noise_scale
            sampled_actions = sampled_actions.numpy() + noise
        else:
            sampled_actions = sampled_actions.numpy()

        # クリップ
        sampled_actions = np.clip(
            sampled_actions, -self.action_bound, self.action_bound
        )

        # 形状の確認と修正 - 必ず1次元配列で返す
        return sampled_actions.flatten()

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

    def calculate_reward(self, state, next_state, raw_reward, done):
        """MountainCar用カスタム報酬計算"""
        position, velocity = next_state
        prev_position, prev_velocity = state

        # 基本報酬
        reward = raw_reward

        # 位置向上に対する報酬を強化
        if position > prev_position:
            reward += (position - prev_position) * 30

        # 目標近くでの追加ボーナス（より段階的に）
        if position > 0.4:
            reward += 30.0
        if position > 0.45:
            reward += 50.0

        # 成功ボーナスをさらに増加
        if position >= 0.5:
            reward += 200.0

        return reward

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
                reward = self.calculate_reward(
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
                        directory=f"{self.model_dir}/frames/episode_{episode}",
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
                        directory=f"{self.model_dir}/frames/success_{episode}",
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
                    plot_training_progress(
                        self.history,
                        save_path=f"{self.model_dir}/training_progress.png",
                        show=False,
                    )

            # 一定の成功率に達したら早期終了
            if success_rate >= 30 and episode >= 50:
                print(
                    f"目標達成: エピソード {episode} で成功率 {success_rate:.1f}% を達成"
                )
                self.save_models(episode)
                self.save_history()

                # 成功時の画像も保存
                evaluation_frames_dir = f"{self.model_dir}/frames/final_success"
                os.makedirs(evaluation_frames_dir, exist_ok=True)

                # 成功したモデルの実行を可視化
                observation = self.env.reset()
                if isinstance(observation, tuple):
                    observation = observation[0]

                for step in range(200):
                    continuous_action = self.policy(observation, add_noise=False)
                    next_observation, reward, done, _ = self.env.step(continuous_action)

                    save_render_image(
                        self.env,
                        "final",
                        step,
                        observation,
                        reward,
                        continuous_action,
                        False,
                        directory=evaluation_frames_dir,
                    )

                    observation = next_observation

                    if done or observation[0] >= 0.5:
                        # 最後のフレームを保存
                        save_render_image(
                            self.env,
                            "final",
                            step + 1,
                            observation,
                            reward,
                            continuous_action,
                            True,
                            directory=evaluation_frames_dir,
                        )
                        break

                break

        return self.history

    def evaluate(self, episodes=20, render=True, save_frames=True):
        """学習したエージェントの評価"""
        rewards = []
        positions = []
        steps_list = []
        success_count = 0

        # 評価用フレーム保存ディレクトリ
        eval_frames_dir = f"{self.model_dir}/frames/evaluation"
        if save_frames:
            os.makedirs(eval_frames_dir, exist_ok=True)

        for episode in range(episodes):
            try:
                observation = self.env.reset()
                if isinstance(observation, tuple):
                    observation = observation[0]

                total_reward = 0
                max_position = -np.inf
                success = False

                for t in range(1000):  # 長めに実行
                    try:
                        if render:
                            try:
                                self.env.render()
                                import time

                                time.sleep(0.01)  # 表示をゆっくりに
                            except Exception as e:
                                print(f"レンダリングエラー: {e}")
                                render = False  # 以降のレンダリングを無効化

                        # 行動を選択（ノイズなし）
                        continuous_action = self.policy(observation, add_noise=False)

                        # continuous_actionを常に適切な形式に変換
                        if not isinstance(continuous_action, np.ndarray):
                            continuous_action = np.array([continuous_action])
                        elif len(continuous_action.shape) == 0:
                            continuous_action = np.array([continuous_action.item()])

                        # フレーム保存
                        if save_frames and t % 5 == 0:
                            try:
                                from src.common.utils import save_render_image

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
                            except Exception as e:
                                print(f"フレーム保存エラー: {e}")
                                save_frames = False  # 以降のフレーム保存を無効化

                        # 行動を実行
                        observation, reward, done, info = self.env.step(
                            continuous_action
                        )
                        if isinstance(info, tuple):
                            info = info[0]

                        total_reward += reward
                        max_position = max(max_position, observation[0])

                        # 成功判定
                        if observation[0] >= 0.5:
                            success = True

                            # 成功時は最後のフレームを保存
                            if save_frames:
                                try:
                                    from src.common.utils import save_render_image

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
                                except Exception as e:
                                    print(f"最終フレーム保存エラー: {e}")

                        if done:
                            break

                    except Exception as e:
                        print(f"ステップ {t} 実行中にエラーが発生: {e}")
                        break

                rewards.append(total_reward)
                positions.append(max_position)
                steps_list.append(t + 1)

                if success:
                    success_count += 1

                print(
                    f"評価エピソード {episode + 1}: 報酬 = {total_reward:.2f}, 最大位置 = {max_position:.4f}, ステップ数 = {t + 1}, 成功 = {success}"
                )

            except Exception as e:
                print(f"評価エピソード {episode} でエラーが発生: {e}")
                # エラーが発生しても次のエピソードに進む
                continue

        # 少なくとも1つのエピソードが完了したか確認
        if rewards:
            success_rate = success_count / len(rewards) * 100
            print(
                f"評価結果: 平均報酬 = {np.mean(rewards):.2f}, 平均最大位置 = {np.mean(positions):.4f}, 平均ステップ数 = {np.mean(steps_list):.1f}, 成功率 = {success_rate:.1f}%"
            )
        else:
            print("評価中にすべてのエピソードでエラーが発生しました。")
            success_rate = 0

        return rewards, positions, steps_list, success_rate
