import os
import logging
from typing import Dict, List, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike

import torch
from torch import Tensor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class
class Config:
    """
    Configuration class for the agent and environment.

    Parameters:
    -----------
    env_name: str
        Name of the environment.
    num_robots: int
        Number of robots in the environment.
    grid_size: tuple or list
        Size of the grid world (width, height).
    max_speed: float
        Maximum speed of the robots.
    min_distance: float
        Minimum allowed distance between robots.
    optimization_threshold: float
        Threshold for optimization-based planning.
    mapf_discretization: int
        Level of discretization for MAPF methods.
    use_velocity_threshold: bool
        Whether to use the velocity-threshold method.
    flow_theory_constant: float
        Constant for the flow theory algorithm.
    debug_mode: bool
        Whether to enable debug mode for additional logging.
    """

    def __init__(self,
                 env_name: str,
                 num_robots: int,
                 grid_size: Union[list, tuple] = (10, 10),
                 max_speed: float = 1.0,
                 min_distance: float = 0.5,
                 optimization_threshold: float = 0.8,
                 mapf_discretization: int = 5,
                 use_velocity_threshold: bool = True,
                 flow_theory_constant: float = 0.2,
                 debug_mode: bool = False):

        self.env_name = env_name
        self.num_robots = num_robots
        self.grid_size = grid_size
        self.max_speed = max_speed
        self.min_distance = min_distance
        self.optimization_threshold = optimization_threshold
        self.mapf_discretization = mapf_discretization
        self.use_velocity_threshold = use_velocity_threshold
        self.flow_theory_constant = flow_theory_constant
        self.debug_mode = debug_mode

        # Validate and process configuration
        self._validate_configuration()
        self._process_configuration()

        logger.info("Configuration initialized successfully.")

    def _validate_configuration(self):
        """Validate the configuration parameters."""
        if self.num_robots < 1:
            raise ValueError("Number of robots must be at least 1.")

        if not (isinstance(self.grid_size, list) or isinstance(self.grid_size, tuple)):
            raise TypeError("Grid size must be a list or tuple.")
        if len(self.grid_size) != 2:
            raise ValueError("Grid size must be a list or tuple of length 2 (width, height).")
        if any(dim <= 0 for dim in self.grid_size):
            raise ValueError("Grid dimensions must be positive integers.")

        if self.max_speed <= 0:
            raise ValueError("Maximum speed must be a positive number.")

        if self.min_distance < 0:
            raise ValueError("Minimum distance must be a non-negative number.")

        if not 0 <= self.optimization_threshold <= 1:
            raise ValueError("Optimization threshold must be between 0 and 1.")

        if self.mapf_discretization <= 0:
            raise ValueError("MAPF discretization must be a positive integer.")

    def _process_configuration(self):
        """Process and derive additional configuration parameters."""
        # Example: derive a parameter based on environment name
        if "complex_env" in self.env_name:
            self.optimization_threshold *= 0.9  # Adjust threshold for complex environments

        # Example: set a parameter based on robot count
        if self.num_robots > 10:
            self.use_velocity_threshold = False  # Disable velocity threshold for large robot counts

    @property
    def config_dict(self) -> Dict:
        """Get the configuration as a dictionary."""
        return {
            "env_name": self.env_name,
            "num_robots": self.num_robots,
            "grid_size": self.grid_size,
            "max_speed": self.max_speed,
            "min_distance": self.min_distance,
            "optimization_threshold": self.optimization_threshold,
            "mapf_discretization": self.mapf_discretization,
            "use_velocity_threshold": self.use_velocity_threshold,
            "flow_theory_constant": self.flow_theory_constant,
            "debug_mode": self.debug_mode
        }

# Exception classes
class InvalidConfigurationException(Exception):
    """Exception raised for invalid configuration parameters."""
    pass

class RobotConfigurationError(Exception):
    """Exception raised for errors related to robot configuration."""
    pass

# Helper functions
def load_config_from_file(file_path: str) -> Dict:
    """
    Load configuration from a file.

    Parameters:
    -----------
    file_path: str
        Path to the configuration file.

    Returns:
    --------
    dict
        Dictionary containing the configuration parameters.
    """
    try:
        with open(file_path, 'r') as file:
            return eval(file.read())  # Evaluate the contents as a dictionary
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at path: {file_path}")
    except SyntaxError as e:
        raise InvalidConfigurationException(f"Invalid configuration file syntax: {e}")

def create_robot_configs(num_robots: int, grid_size: ArrayLike) -> List[Dict]:
    """
    Create configurations for multiple robots.

    Parameters:
    -----------
    num_robots: int
        Number of robots to create configurations for.
    grid_size: array_like
        Size of the grid world (width, height).

    Returns:
    --------
    list of dict
        List of configurations for each robot.
    """
    if num_robots < 1:
        raise RobotConfigurationError("Number of robots must be at least 1.")

    robot_configs = []
    for i in range(num_robots):
        x_pos = np.random.uniform(0, grid_size[0])
        y_pos = np.random.uniform(0, grid_size[1])
        robot_config = {"id": i, "position": (x_pos, y_pos)}
        robot_configs.append(robot_config)

    return robot_configs

# Main function
def main():
    # Example usage of the configuration class
    config_file = "agent_config.conf"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    config_dict = load_config_from_file(config_file)
    config = Config(**config_dict)

    # Example of creating robot configurations
    num_robots = config.num_robots
    grid_size = config.grid_size
    robot_configs = create_robot_configs(num_robots, grid_size)
    for robot_config in robot_configs:
        print(robot_config)

    # Example of accessing configuration values
    print(f"Environment name: {config.env_name}")
    print(f"Number of robots: {config.num_robots}")
    print(f"Grid size: {config.grid_size}")
    print(f"Optimization threshold: {config.optimization_threshold}")

    # Example of modifying a configuration value
    config.debug_mode = True
    print(f"Debug mode: {config.debug_mode}")

    # Example of using configuration with other components
    # ...

if __name__ == "__main__":
    main()