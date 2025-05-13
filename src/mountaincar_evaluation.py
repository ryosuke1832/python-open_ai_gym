import argparse
import os
import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="MountainCar DDPGモデル評価スクリプト")
parser.add_argument(
    "--model_dir",
    type=str,
    default="saved_models",
    help="モデルが保存されているディレクトリ",
)
parser.add_argument(
    "--episode", type=int, default=273, help="読み込むモデルのエピソード番号"
)
parser.add_argument("--episodes", type=int, default=20, help="評価するエピソード数")
parser.add_argument("--no_render", action="store_true", help="レンダリングを無効にする")
args = parser.parse_args()


class DDPGAgent:
    """簡易版DDPGエージェント（評価用）"""

    def __init__(self, env, model_dir="saved_models"):
        self.env = env
        self.state_dim = 2  # MountainCarは2次元状態
        self.action_dim = 1  # DDPGは連続値を出力
        self.action_bound = 1.0  # 行動の境界値
        self.model_dir = model_dir

        # アクターネットワークの構築
        self.actor_model = self.get_actor_model()

        # 状態正規化用のパラメータ
        self.state_mean = np.array([-0.6, 0.0])  # 位置と速度の平均
        self.state_std = np.array([0.5, 0.035])  # 位置と速度の標準偏差

    def get_actor_model(self):
        """アクターネットワークの定義"""
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(self.action_dim, activation="tanh")(x)
        outputs = outputs * self.action_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def normalize_state(self, state):
        """状態の正規化"""
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def policy(self, state, add_noise=False):
        """行動選択ポリシー"""
        # 状態の正規化
        norm_state = self.normalize_state(state.reshape(1, -1))
        sampled_actions = self.actor_model(norm_state)
        return sampled_actions.numpy()[0]

    def continuous_to_discrete(self, continuous_action):
        """連続値を離散行動に変換"""
        # -1~1の範囲から0, 1, 2に変換
        if continuous_action < -0.33:
            return 0  # 左に押す
        elif continuous_action > 0.33:
            return 2  # 右に押す
        else:
            return 1  # 押さない

    def load_models(self, episode):
        """モデルの読み込み"""
        try:
            actor_path = f"{self.model_dir}/actor_episode_{episode}.h5"
            if os.path.exists(actor_path):
                self.actor_model.load_weights(actor_path)
                print(f"モデルを読み込みました: エピソード {episode}")
                return True
            else:
                print(f"モデルが見つかりません: {actor_path}")
                return False
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False

    def evaluate(self, episodes=20, render=True, delay=0.01):
        """学習したエージェントの評価"""
        rewards = []
        positions = []
        steps_list = []
        success_count = 0

        for episode in range(episodes):
            observation = self.env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]

            total_reward = 0
            max_position = -np.inf
            success = False
            actions_taken = {0: 0, 1: 0, 2: 0}  # 各行動の選択回数

            for t in range(200):  # 標準のステップ数
                if render:
                    self.env.render()
                    time.sleep(delay)  # 表示を遅くする

                # 学習済みモデルから行動を選択
                continuous_action = self.policy(observation)[0]
                discrete_action = self.continuous_to_discrete(continuous_action)
                actions_taken[discrete_action] += 1

                # 行動を実行
                observation, reward, done, info = self.env.step(discrete_action)
                if isinstance(info, tuple):
                    info = info[0]

                total_reward += reward
                max_position = max(max_position, observation[0])

                # 成功判定
                if observation[0] >= 0.5:
                    success = True

                if done:
                    break

            rewards.append(total_reward)
            positions.append(max_position)
            steps_list.append(t + 1)

            if success:
                success_count += 1

            print(f"評価エピソード {episode + 1}:")
            print(f"  報酬 = {total_reward:.2f}")
            print(f"  最大位置 = {max_position:.4f}")
            print(f"  ステップ数 = {t + 1}")
            print(f"  成功 = {success}")
            print(
                f"  行動選択: 左={actions_taken[0]}, 無し={actions_taken[1]}, 右={actions_taken[2]}"
            )
            print()

        success_rate = success_count / episodes * 100
        print("評価結果:")
        print(f"  平均報酬 = {np.mean(rewards):.2f}")
        print(f"  平均最大位置 = {np.mean(positions):.4f}")
        print(f"  平均ステップ数 = {np.mean(steps_list):.1f}")
        print(f"  成功率 = {success_rate:.1f}%")
        return rewards, positions, steps_list, success_rate


def main():
    try:
        # まず新しいバージョンのGymの方法を試す
        env = gym.make(
            "MountainCar-v0", render_mode="human" if not args.no_render else None
        )
    except (TypeError, ValueError):
        # 古いバージョンのGymの場合
        env = gym.make("MountainCar-v0")

    # DDPGエージェントの作成
    agent = DDPGAgent(env, model_dir=args.model_dir)

    # モデルの読み込み
    if not agent.load_models(args.episode):
        print("モデルの読み込みに失敗しました。終了します。")
        return

    # 評価
    print(f"\n学習したDDPGモデル（エピソード {args.episode}）の評価を開始...")
    try:
        agent.evaluate(episodes=args.episodes, render=not args.no_render)
    except KeyboardInterrupt:
        print("\n評価が中断されました")
    finally:
        env.close()


if __name__ == "__main__":
    main()
