import os

import numpy as np


class BaseAgent:
    """強化学習エージェントの基本クラス"""

    def __init__(self, env, state_dim, action_dim, model_dir="saved_models"):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dir = model_dir

        # ディレクトリがなければ作成
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 学習履歴の初期化
        self.history = {
            "rewards": [],
            "avg_rewards": [],
            "positions": [],
            "steps": [],
            "successes": [],
        }

    def policy(self, state, add_noise=True):
        """行動選択ポリシー（オーバーライド必要）"""
        raise NotImplementedError

    def remember(self, state, action, reward, next_state, done):
        """経験をバッファに追加（オーバーライド必要）"""
        raise NotImplementedError

    def learn(self):
        """学習ステップ（オーバーライド必要）"""
        raise NotImplementedError

    def calculate_reward(self, state, next_state, raw_reward, done):
        """報酬の計算（環境別にオーバーライド）"""
        return raw_reward

    def save_models(self, episode):
        """モデルの保存（オーバーライド必要）"""
        raise NotImplementedError

    def load_models(self, episode):
        """モデルの読み込み（オーバーライド必要）"""
        raise NotImplementedError

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

    def train(self, episodes=1000, max_steps=1000, save_interval=100, start_episode=0):
        """エージェントの学習（基本実装）"""
        raise NotImplementedError

    def evaluate(self, episodes=20, render=True, save_frames=True):
        """エージェントの評価（基本実装）"""
        raise NotImplementedError
