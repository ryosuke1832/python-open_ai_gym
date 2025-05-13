import numpy as np


class ReplayBuffer:
    """経験リプレイバッファの汎用実装"""

    def __init__(
        self, buffer_capacity=10000, batch_size=64, state_dim=None, action_dim=None
    ):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        # メモリの初期化
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim))
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
