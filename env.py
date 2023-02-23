from typing import Optional, List, Tuple
import numpy as np
from gymnasium.envs.box2d import LunarLander
    
from algorithms import Policy, Trajectory

class ActionOffsetLunarLander(LunarLander):
    def __init__(
        self,
        task_params: np.ndarray,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        super().__init__(render_mode="rgb_array", continuous=True, gravity=gravity, enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
        
        # Store randomized task-specific parameters
        self.task_params = task_params
        
        # State variable to store action offset for current task
        self.action_offset = self.task_params[0]
        
    def reset(self, task_index: int, *args, **kwargs):
        self.action_offset = self.task_params[task_index]
        return super().reset(*args, **kwargs)
        
    def step(self, action):
        return super().step(action.astype(np.float64) + self.action_offset)
        


def collect_trajectories(env: ActionOffsetLunarLander, policy: Policy, 
                         task_indices: List[int], num_episodes: int,
                         max_episode_length: int = 400, render: bool = False
                         ) -> Tuple[
                                List[List[Trajectory]],
                                dict,
                                Optional[List[List[np.ndarray]]]
                            ]:
    """_summary_

    Args:
        env (ActionOffsetLunarLander): _description_
        policy (Policy): _description_
        task_indices (List[int]): _description_
        num_episodes (int): _description_
        render (bool): Whether to render frames for each task.

    Returns:
        List[List[Trajectory]]: List (over tasks) of list (over episodes) of trajectories.
        dict: Metrics to log.
        Optional[List[List[np.ndarray]]]: List of rendered frames for each task.
    """
    trajectories = []
    if render:
        frames = []
    else:
        frames = None
        
    for task_index in task_indices:
        trajectories.append([])
        if render:
            frames.append([])
        for episode_index in range(num_episodes):
            states = []
            actions = []
            rewards = []
            
            policy.reset(task_index)
            s, _ = env.reset(task_index)
            terminated = False
            truncated = False
            episode_length = 0
            if render:
                frames[-1].append(env.render())
            
            # Save initial state
            states.append(s)
            
            while not (terminated or truncated):
                a = policy.get_action(s)
                next_s, r, terminated, truncated, info = env.step(a)
                episode_length += 1
                
                # Artificially truncate if over episode max length
                if episode_length >= max_episode_length:
                    truncated = True
                
                if render:
                    frames[-1].append(env.render())
                
                # Add to saved trajectory info
                actions.append(a)
                rewards.append(r)
                states.append(next_s)
                
                # Provide feedback to policy
                policy.update_memory(s, a, r, next_s)
                
                # Prepare for next step
                s = next_s
            
            trajectories[-1].append(Trajectory(task_index, states, actions, rewards, terminated=terminated))
            
    # TODO: Compute metrics
    metrics = {
        "mean_return": np.mean([traj.rewards.sum() for task_trajs in trajectories for traj in task_trajs]),
        "mean_ep_len": np.mean([len(traj) for task_trajs in trajectories for traj in task_trajs])
    }
        
    return trajectories, metrics, frames