# src/agents/__init__.py

from src.agents.base_agent import BaseAgent
from src.agents.ddpg_agent import DDPGAgent
from src.agents.dqn_agent import DQNAgent

# PPOエージェントを追加した場合
# from src.agents.ppo_agent import PPOAgent

__all__ = [
    "BaseAgent",
    "DDPGAgent",
    "dqn_agent",  # DDPGエージェントを追加した場合
    # 'PPOAgent'  # PPOを追加した場合
]
