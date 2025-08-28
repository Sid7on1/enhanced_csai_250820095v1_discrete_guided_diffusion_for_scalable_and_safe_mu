import logging
import numpy as np
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from scipy.spatial import distance
from scipy.stats import norm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
class RewardConfig(Enum):
    VELOCITY_THRESHOLD = 1
    FLOW_THEORY = 2

@dataclass
class RewardConfigData:
    reward_config: RewardConfig
    velocity_threshold: float
    flow_theory_alpha: float
    flow_theory_beta: float

class RewardSystem:
    def __init__(self, config: RewardConfigData):
        self.config = config
        self.velocity_threshold = config.velocity_threshold
        self.flow_theory_alpha = config.flow_theory_alpha
        self.flow_theory_beta = config.flow_theory_beta

    def calculate_reward(self, agent_positions: List[Tuple[float, float]], goal_positions: List[Tuple[float, float]]) -> Dict[str, float]:
        rewards = {}
        for agent_position, goal_position in zip(agent_positions, goal_positions):
            agent_x, agent_y = agent_position
            goal_x, goal_y = goal_position
            distance_to_goal = distance.euclidean((agent_x, agent_y), (goal_x, goal_y))
            if self.config == RewardConfig.VELOCITY_THRESHOLD:
                velocity = np.linalg.norm(np.array([agent_x, agent_y]) - np.array([goal_x, goal_y]))
                reward = -velocity if velocity > self.velocity_threshold else 0
            elif self.config == RewardConfig.FLOW_THEORY:
                reward = -norm.pdf(distance_to_goal, loc=self.flow_theory_alpha, scale=self.flow_theory_beta)
            rewards[f"agent_{agent_positions.index((agent_x, agent_y))}"] = reward
        return rewards

class RewardShaper(ABC):
    @abstractmethod
    def shape_reward(self, reward: float) -> float:
        pass

class VelocityThresholdRewardShaper(RewardShaper):
    def shape_reward(self, reward: float) -> float:
        return reward * 10

class FlowTheoryRewardShaper(RewardShaper):
    def shape_reward(self, reward: float) -> float:
        return reward * 5

class RewardCalculator:
    def __init__(self, reward_system: RewardSystem, reward_shaper: RewardShaper):
        self.reward_system = reward_system
        self.reward_shaper = reward_shaper

    def calculate_reward(self, agent_positions: List[Tuple[float, float]], goal_positions: List[Tuple[float, float]]) -> float:
        rewards = self.reward_system.calculate_reward(agent_positions, goal_positions)
        shaped_reward = sum(rewards.values())
        shaped_reward = self.reward_shaper.shape_reward(shaped_reward)
        return shaped_reward

# Example usage
if __name__ == "__main__":
    config = RewardConfigData(
        reward_config=RewardConfig.FLOW_THEORY,
        velocity_threshold=1.0,
        flow_theory_alpha=0.5,
        flow_theory_beta=0.1
    )
    reward_system = RewardSystem(config)
    reward_shaper = FlowTheoryRewardShaper()
    reward_calculator = RewardCalculator(reward_system, reward_shaper)
    agent_positions = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    goal_positions = [(3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]
    reward = reward_calculator.calculate_reward(agent_positions, goal_positions)
    logger.info(f"Reward: {reward}")