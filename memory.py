import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32
EXPERIENCE_REPLAY_ALPHA = 0.6
EXPERIENCE_REPLAY_BETA = 0.4

class MemoryType(Enum):
    """Enum for memory types"""
    REPLAY = 1
    PRIORITIZED = 2

class Memory(ABC):
    """Abstract base class for memory"""
    def __init__(self, capacity: int, memory_type: MemoryType):
        self.capacity = capacity
        self.memory_type = memory_type
        self.memory = deque(maxlen=capacity)

    @abstractmethod
    def add_experience(self, experience: Dict):
        """Add experience to memory"""
        pass

    @abstractmethod
    def sample_batch(self) -> Dict:
        """Sample batch from memory"""
        pass

class ReplayMemory(Memory):
    """Replay memory implementation"""
    def __init__(self, capacity: int):
        super().__init__(capacity, MemoryType.REPLAY)

    def add_experience(self, experience: Dict):
        """Add experience to replay memory"""
        self.memory.append(experience)

    def sample_batch(self) -> Dict:
        """Sample batch from replay memory"""
        batch = {}
        if len(self.memory) >= BATCH_SIZE:
            batch_size = BATCH_SIZE
        else:
            batch_size = len(self.memory)
        batch['states'] = np.array([x['state'] for x in random.sample(self.memory, batch_size)])
        batch['actions'] = np.array([x['action'] for x in random.sample(self.memory, batch_size)])
        batch['rewards'] = np.array([x['reward'] for x in random.sample(self.memory, batch_size)])
        batch['next_states'] = np.array([x['next_state'] for x in random.sample(self.memory, batch_size)])
        batch['done'] = np.array([x['done'] for x in random.sample(self.memory, batch_size)])
        return batch

class PrioritizedMemory(Memory):
    """Prioritized memory implementation"""
    def __init__(self, capacity: int):
        super().__init__(capacity, MemoryType.PRIORITIZED)
        self.priorities = deque(maxlen=capacity)

    def add_experience(self, experience: Dict):
        """Add experience to prioritized memory"""
        self.memory.append(experience)
        self.priorities.append(1.0)

    def sample_batch(self) -> Dict:
        """Sample batch from prioritized memory"""
        batch = {}
        if len(self.memory) >= BATCH_SIZE:
            batch_size = BATCH_SIZE
        else:
            batch_size = len(self.memory)
        batch['states'] = np.array([x['state'] for x in random.sample(self.memory, batch_size)])
        batch['actions'] = np.array([x['action'] for x in random.sample(self.memory, batch_size)])
        batch['rewards'] = np.array([x['reward'] for x in random.sample(self.memory, batch_size)])
        batch['next_states'] = np.array([x['next_state'] for x in random.sample(self.memory, batch_size)])
        batch['done'] = np.array([x['done'] for x in random.sample(self.memory, batch_size)])
        batch['priorities'] = np.array([self.priorities[i] for i in random.sample(range(len(self.memory)), batch_size)])
        return batch

class ExperienceReplay:
    """Experience replay class"""
    def __init__(self, memory_type: MemoryType):
        self.memory_type = memory_type
        if memory_type == MemoryType.REPLAY:
            self.memory = ReplayMemory(MEMORY_CAPACITY)
        elif memory_type == MemoryType.PRIORITIZED:
            self.memory = PrioritizedMemory(MEMORY_CAPACITY)

    def add_experience(self, experience: Dict):
        """Add experience to experience replay"""
        self.memory.add_experience(experience)

    def sample_batch(self) -> Dict:
        """Sample batch from experience replay"""
        return self.memory.sample_batch()

class ExperienceReplayBuffer:
    """Experience replay buffer class"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add_experience(self, experience: Dict):
        """Add experience to experience replay buffer"""
        self.buffer.append(experience)

    def sample_batch(self) -> Dict:
        """Sample batch from experience replay buffer"""
        batch_size = min(BATCH_SIZE, len(self.buffer))
        batch = {}
        batch['states'] = np.array([x['state'] for x in random.sample(self.buffer, batch_size)])
        batch['actions'] = np.array([x['action'] for x in random.sample(self.buffer, batch_size)])
        batch['rewards'] = np.array([x['reward'] for x in random.sample(self.buffer, batch_size)])
        batch['next_states'] = np.array([x['next_state'] for x in random.sample(self.buffer, batch_size)])
        batch['done'] = np.array([x['done'] for x in random.sample(self.buffer, batch_size)])
        return batch

class ExperienceReplayAgent:
    """Experience replay agent class"""
    def __init__(self, memory_type: MemoryType):
        self.memory_type = memory_type
        self.experience_replay = ExperienceReplay(memory_type)
        self.experience_replay_buffer = ExperienceReplayBuffer(MEMORY_CAPACITY)

    def add_experience(self, experience: Dict):
        """Add experience to experience replay agent"""
        self.experience_replay_buffer.add_experience(experience)
        self.experience_replay.add_experience(experience)

    def sample_batch(self) -> Dict:
        """Sample batch from experience replay agent"""
        return self.experience_replay.sample_batch()

def main():
    # Create experience replay agent
    agent = ExperienceReplayAgent(MemoryType.REPLAY)

    # Add experience to experience replay agent
    experience = {
        'state': np.array([1, 2, 3]),
        'action': np.array([4, 5, 6]),
        'reward': 10,
        'next_state': np.array([7, 8, 9]),
        'done': False
    }
    agent.add_experience(experience)

    # Sample batch from experience replay agent
    batch = agent.sample_batch()
    print(batch)

if __name__ == '__main__':
    main()