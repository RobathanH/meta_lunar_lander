from typing import Tuple, Optional
import os
import json
import numpy as np
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from algorithms.base import Trajectory

'''
Simple replay buffer, samples shuffled data.
Stores tuples concatenated into torch Tensors:
state, action, reward, next_state, done_flag
'''
class ExpBuffer:
    def __init__(self, capacity: int, obs_size: int, act_size: int, load_dir: Optional[str] = None):
        self.capacity = capacity
        self.obs_size = obs_size
        self.act_size = act_size
        
        # Store data concatenated in tensor array (s, a, r, s', done_flag)
        if load_dir is None:
            self.buffer = torch.zeros((capacity, obs_size + act_size + 1 + obs_size + 1))
            self.next_ind = 0
            self.full = False
        else:
            self.buffer = torch.load(os.path.join(load_dir, "exp_buffer.pt"))
            assert self.buffer.shape == (capacity, obs_size + act_size + 1 + obs_size + 1)
            with open(os.path.join(load_dir, "exp_buffer_state.json"), "r") as fp:
                state = json.load(fp)
                self.next_ind = state["next_ind"]
                self.full = state["full"]
        
    def __len__(self) -> int:
        return self.capacity if self.full else self.next_ind
        
    def clear(self) -> None:
        self.next_ind = 0
        self.full = False
    
    def save(self, output_dir: str) -> None:
        torch.save(self.buffer, os.path.join(output_dir, "exp_buffer.pt"))
        with open(os.path.join(output_dir, "exp_buffer_state.json"), "w") as fp:
            json.dump({
                "next_ind": self.next_ind,
                "full": self.full
            }, fp)
        
    def add_trajectory(self, traj: Trajectory) -> None:
        # Concatenate tuples
        new_samples = torch.from_numpy(np.concatenate([
            traj.states, traj.actions, traj.rewards, traj.next_states, traj.done_mask
        ], axis=1))
        
        # Add to buffer, considering wrapping around to start position
        samples_remaining = len(new_samples)
        while samples_remaining:
            samples_before_wrap = min(samples_remaining, self.capacity - self.next_ind)
            self.buffer[self.next_ind : self.next_ind + samples_before_wrap] = new_samples[len(new_samples) - samples_remaining : len(new_samples) - samples_remaining + samples_before_wrap]
            
            # Update counters
            self.next_ind += samples_before_wrap
            samples_remaining -= samples_before_wrap
        
            # Potentially wrap index back to buffer start
            if self.next_ind == self.capacity:
                self.next_ind = 0
                self.full = True
                
    def sample(self, batch_size: int) -> torch.Tensor:
        indices = np.random.choice(len(self), size=batch_size, replace=True)
        batch = self.buffer[indices]
        return batch
    


'''
Multi-task exp buffer. Maintains a separate exp buffer for each task,
but still samples tuples in a shuffled format (Trajectories don't stay together)
'''
class MultiTaskExpBuffer:
    def __init__(self, num_train_tasks: int, capacity: int, obs_size: int, act_size: int, load_dir: Optional[str] = None):
        self.num_train_tasks = num_train_tasks
        self.capacity = capacity
        self.obs_size = obs_size
        self.act_size = act_size
        
        # Store data concatenated in tensor array (s, a, r, s', done_flag)
        if load_dir is None:
            self.buffer = torch.zeros((num_train_tasks, capacity, obs_size + act_size + 1 + obs_size + 1))
            self.next_ind = [0 for _ in range(num_train_tasks)]
            self.full = [False for _ in range(num_train_tasks)]
        else:
            self.buffer = torch.load(os.path.join(load_dir, "exp_buffer.pt"))
            assert self.buffer.shape == (num_train_tasks, capacity, obs_size + act_size + 1 + obs_size + 1)
            with open(os.path.join(load_dir, "exp_buffer_state.json"), "r") as fp:
                state = json.load(fp)
                self.next_ind = state["next_ind"]
                self.full = state["full"]
                
    def len(self, task_index: int) -> int:
        return self.capacity if self.full[task_index] else self.next_ind[task_index]
    
    def clear(self) -> None:
        self.next_ind = [0 for _ in range(self.num_train_tasks)]
        self.full = [False for _ in range(self.num_train_tasks)]
        
    def save(self, output_dir: str) -> None:
        torch.save(self.buffer, os.path.join(output_dir, "exp_buffer.pt"))
        with open(os.path.join(output_dir, "exp_buffer_state.json"), "w") as fp:
            json.dump({
                "next_ind": self.next_ind,
                "full": self.full
            }, fp)
            
    def add_trajectory(self, task_index: int, traj: Trajectory, max_length: Optional[int] = None) -> None:
        # Concatenate tuples
        if max_length is None:
            new_samples = torch.from_numpy(np.concatenate([
                traj.states, traj.actions, traj.rewards, traj.next_states, traj.done_mask
            ], axis=1))
        else:
            new_samples = torch.from_numpy(np.concatenate([
                traj.states[:max_length], traj.actions[:max_length], traj.rewards[:max_length], traj.next_states[:max_length], traj.done_mask[:max_length]
            ], axis=1))
        
        # Add to buffer, considering wrapping around to start position
        samples_remaining = len(new_samples)
        while samples_remaining:
            samples_before_wrap = min(samples_remaining, self.capacity - self.next_ind[task_index])
            self.buffer[task_index, self.next_ind[task_index] : self.next_ind[task_index] + samples_before_wrap] = new_samples[len(new_samples) - samples_remaining : len(new_samples) - samples_remaining + samples_before_wrap]
            
            # Update counters
            self.next_ind[task_index] += samples_before_wrap
            samples_remaining -= samples_before_wrap
        
            # Potentially wrap index back to buffer start
            if self.next_ind[task_index] == self.capacity:
                self.next_ind[task_index] = 0
                self.full[task_index] = True
                
    def sample(self, task_index: int, batch_size: int) -> torch.Tensor:
        indices = np.random.choice(self.len(task_index), size=batch_size, replace=True)
        batch = self.buffer[task_index, indices]
        return batch