import gym


class CartPoleEnv:
    """CartPole環境のラッパークラス"""

    def __init__(self):
        # 環境の作成
        self.env = gym.make("CartPole-v0")
        self.state_dim = 4  # カートの位置、速度、ポールの角度、ポールの角速度
        self.action_dim = 2  # 左右2つの離散行動

        # 報酬設計のパラメータ
        self.reward_scale = 1.0

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        """環境のリセット"""
        return self.env.reset()

    def step(self, action):
        """環境の1ステップ実行"""
        return self.env.step(action)

    def render(self, mode="human"):
        """環境のレンダリング"""
        return self.env.render(mode=mode)

    def close(self):
        """環境のクローズ"""
        return self.env.close()

    def calculate_reward(self, state, next_state, raw_reward, done):
        """カスタム報酬関数（必要に応じてカスタマイズ）"""
        # CartPoleのデフォルト報酬はすでに良く設計されているので、
        # そのまま使用するかわずかな修正にとどめる

        # 失敗時のペナルティを追加
        if done and raw_reward == 0:  # エピソード途中で失敗
            return -10.0

        # 長く持続するほど報酬を増やす
        return raw_reward * self.reward_scale
