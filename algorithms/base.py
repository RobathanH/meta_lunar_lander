from abc import ABC, abstractmethod
from typing import List, Optional
import os
import json
import numpy as np



'''
Simple class to store a rollout trajectory
Properties:
    states: 
    actions:
    rewards:
    next_states:
'''
class Trajectory:
    def __init__(self, task_index: int, states: List[np.ndarray], actions: List[np.ndarray], rewards: List[float]):
        """
        Args:
            task_index (int): Index of the task used to collect this trajectory.
            states (List[np.ndarray]): List of states encountered, including the final ending state
            actions (List[np.ndarray]): List of actions taken. Should be 1 shorter than states list.
            rewards (List[float]): List of rewards received. Should be 1 shorter than states list.
        """
        self.task_index = task_index
        self.states = np.array(states[:-1])
        self.actions = np.array(actions)
        self.rewards = np.array(rewards).reshape(-1, 1)
        self.next_states = np.array(states[1:])
        
    def __len__(self) -> int:
        return len(self.states)

'''
Base interface for policies, which provide actions and rollouts from the current trained network
'''
class Policy(ABC):
    @abstractmethod
    def reset(self, task_index: int) -> None:
        """Reset the policy for a new episode/task.
        This would mean resetting the hidden state for an RNN,
        resetting to initial parameters for a MAML algorithm,
        or nothing for historyless policies in vanilla RL.
        
        Args:
            task_index (int): index for the coming task. Shouldn't be used except for
            policies which cheat with extra information, like NoOffsetSAC.
        """
        pass
    
    @abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Choose action from state.
        """
        pass
    
    @abstractmethod
    def update_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        """Save the most recent transition from the current trajectory.
        This would let an RNN update its hidden state, or other methods
        update their action offset estimate.

        Args:
            state (np.ndarray): _description_
            action (np.ndarray): _description_
            reward (float): _description_
            next_state (np.ndarray): _description_
        """
        pass


'''
Base interface for implementing RL algorithms
'''
class Trainer(ABC):
    def __init__(self, config: dict, load_dir: Optional[str] = None):
        """
        Args:
            config (dict): Config dict for the current experiment.
            load_dir (Optional[str]): Previous output dir from which to load task parameters,
            network weights and exp buffer (if algo uses offline data). Default is None.
        """
        self.config = config
        
        # Create or load task parameters
        if load_dir is None:
            self.task_params = np.array([
                config["action_offset_magnitude"] * np.random.uniform(-1, 1, size=config["action_size"])
                for _ in range(config["num_train_tasks"] + config["num_test_tasks"])
            ])
        else:
            self.task_params = np.load(os.path.join(load_dir, "task_params.npy"))
            
        # Create or load state variables
        if load_dir is None:
            self.trainer_state = {
                "global_step": 0
            }
        else:
            with open(os.path.join(load_dir, "trainer_state.json"), "r") as fp:
                self.trainer_state = json.load(fp)
        
    def save(self, output_dir: str) -> None:
        """Saves the stored config and task parameters to fixed filenames within the save folder.
        Subclasses should also save network weights and potentially exp buffer contents (if alg uses offline data).

        Args:
            output_dir (str): Directory containing outputs of a particular experiment.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config (not that it changes, but good to make sure it's there to check what a run was using)
        with open(os.path.join(output_dir, "config.json"), "w") as fp:
            json.dump(self.config, fp)
        
        # Save task parameters (these also don't change, but should be saved for if we resume this run later)
        np.save(os.path.join(output_dir, "task_params.npy"), self.task_params)
        
        # Save trainer state (right now this just saves the total number of training steps, allowing logs to resume without overriding prev run)
        with open(os.path.join(output_dir, "trainer_state.json"), "w") as fp:
            json.dump(self.trainer_state, fp)
        
    def current_policy(self) -> Policy:
        """Current most-trained policy for rollouts.
        """
        raise NotImplementedError
        
    def train_step(self, task_indices: List[int], trajectories: List[List[Trajectory]]) -> dict:
        """_summary_

        Args:
            task_indices (List[int]): _description_
            trajectories (List[List[Trajectory]]): _description_

        Returns:
            dict: Any training metrics that should be logged.
        """
        # Update trainer state
        self.trainer_state["global_step"] += 1
        return {}