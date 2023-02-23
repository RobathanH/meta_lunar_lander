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
        samples_remaining = len(traj)
        while samples_remaining:
            samples_before_wrap = min(samples_remaining, self.capacity - self.next_ind)
            self.buffer[self.next_ind : self.next_ind + samples_before_wrap] = new_samples[len(traj) - samples_remaining : len(traj) - samples_remaining + samples_before_wrap]
            
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