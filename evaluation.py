import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EvaluationMetric(Enum):
    """Enum for evaluation metrics"""
    VELOCITY_THRESHOLD = 1
    FLOW_THEORY = 2

@dataclass
class AgentEvaluationResult:
    """Data class for agent evaluation result"""
    metric: EvaluationMetric
    value: float
    error: str = None

class EvaluationException(Exception):
    """Custom exception for evaluation errors"""
    pass

class AgentEvaluator(ABC):
    """Abstract base class for agent evaluators"""
    @abstractmethod
    def evaluate(self, agent_data: Dict) -> AgentEvaluationResult:
        """Evaluate agent performance"""
        pass

class VelocityThresholdEvaluator(AgentEvaluator):
    """Evaluator for velocity threshold metric"""
    def __init__(self, threshold: float):
        self.threshold = threshold

    def evaluate(self, agent_data: Dict) -> AgentEvaluationResult:
        """Evaluate agent performance based on velocity threshold"""
        try:
            velocity = agent_data['velocity']
            if velocity > self.threshold:
                return AgentEvaluationResult(EvaluationMetric.VELOCITY_THRESHOLD, velocity)
            else:
                return AgentEvaluationResult(EvaluationMetric.VELOCITY_THRESHOLD, velocity, "Velocity below threshold")
        except KeyError as e:
            raise EvaluationException("Missing key in agent data") from e

class FlowTheoryEvaluator(AgentEvaluator):
    """Evaluator for flow theory metric"""
    def __init__(self, flow_rate: float):
        self.flow_rate = flow_rate

    def evaluate(self, agent_data: Dict) -> AgentEvaluationResult:
        """Evaluate agent performance based on flow theory"""
        try:
            flow = agent_data['flow']
            if flow > self.flow_rate:
                return AgentEvaluationResult(EvaluationMetric.FLOW_THEORY, flow)
            else:
                return AgentEvaluationResult(EvaluationMetric.FLOW_THEORY, flow, "Flow below rate")
        except KeyError as e:
            raise EvaluationException("Missing key in agent data") from e

class AgentEvaluationService:
    """Service class for agent evaluation"""
    def __init__(self, evaluators: List[AgentEvaluator]):
        self.evaluators = evaluators

    def evaluate_agent(self, agent_data: Dict) -> List[AgentEvaluationResult]:
        """Evaluate agent performance using multiple evaluators"""
        results = []
        for evaluator in self.evaluators:
            try:
                result = evaluator.evaluate(agent_data)
                results.append(result)
            except EvaluationException as e:
                logging.error(f"Error evaluating agent: {e}")
        return results

class AgentEvaluationConfig:
    """Configuration class for agent evaluation"""
    def __init__(self, velocity_threshold: float, flow_rate: float):
        self.velocity_threshold = velocity_threshold
        self.flow_rate = flow_rate

def create_evaluators(config: AgentEvaluationConfig) -> List[AgentEvaluator]:
    """Create evaluators based on configuration"""
    evaluators = [
        VelocityThresholdEvaluator(config.velocity_threshold),
        FlowTheoryEvaluator(config.flow_rate)
    ]
    return evaluators

def main():
    # Create configuration
    config = AgentEvaluationConfig(velocity_threshold=10.0, flow_rate=5.0)

    # Create evaluators
    evaluators = create_evaluators(config)

    # Create evaluation service
    service = AgentEvaluationService(evaluators)

    # Create agent data
    agent_data = {'velocity': 15.0, 'flow': 7.0}

    # Evaluate agent
    results = service.evaluate_agent(agent_data)

    # Print results
    for result in results:
        logging.info(f"Metric: {result.metric}, Value: {result.value}, Error: {result.error}")

if __name__ == "__main__":
    main()