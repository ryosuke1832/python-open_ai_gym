# src/agents/__init__.py

from src.agents.base_agent import BaseAgent
from src.agents.ddpg_agent import DDPGAgent

# PPOエージェントを追加した場合
# from src.agents.ppo_agent import PPOAgent

__all__ = [
    "BaseAgent",
    "DDPGAgent",
    # 'PPOAgent'  # PPOを追加した場合
]
