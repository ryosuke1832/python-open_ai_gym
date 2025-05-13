import gym


class MountainCarContinuousEnv:
    """MountainCar環境のラッパークラス"""

    def __init__(self):
        # 環境の作成
        self.env = gym.make("MountainCarContinuous-v0")
        self.state_dim = 2  # 位置と速度
        self.action_dim = 1  # 連続的な力
        self.action_bound = 1.0

        # 報酬設計のパラメータ
        self.reward_scale = 30.0
        self.goal_position = 0.5

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
        """カスタム報酬関数"""
        position, velocity = next_state
        prev_position, prev_velocity = state

        # 基本報酬
        reward = raw_reward

        # 位置向上に対する報酬
        if position > prev_position:
            reward += (position - prev_position) * self.reward_scale

        # 目標近くでの追加ボーナス
        if position > 0.4:
            reward += 30.0
        if position > 0.45:
            reward += 50.0

        # 成功ボーナス
        if position >= self.goal_position:
            reward += 200.0

        return reward
