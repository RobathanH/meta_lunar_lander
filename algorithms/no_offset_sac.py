import os
import json
from typing import Optional, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from .base import Trainer, Policy, Trajectory
from networks import MLP, GaussianPolicyNet
from exp_buffer import ExpBuffer

'''
Simple agent which solves the original lunar lander problem. To work
within this framework, this class uses knowledge of the true task params
to manually correct for task action offsets, so the underlying network
only sees data as it would look in the no-offset environment.
'''


class OffsetCorrectedGaussianPolicy(Policy):
    def __init__(self, policy_net: GaussianPolicyNet, task_params: np.ndarray):
        self.policy_net = policy_net
        self.task_params = task_params
        self.current_task_index = 0
        
    def reset(self, task_index: int) -> None:
        self.current_task_index = task_index
        
    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.from_numpy(state).to(DEVICE)
        action = self.policy_net.get_action(state).cpu().numpy()
        corrected_action = action - self.task_params[self.current_task_index]
        return corrected_action
    
    def update_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        pass


class NoOffsetSAC(Trainer):
    def __init__(self, config: dict, load_dir: Optional[str] = None):
        """
        Args:
            config (dict): Config dict for the current experiment.
            load_dir (Optional[str]): Previous output dir from which to load task parameters,
            network weights and exp buffer (if algo uses offline data). Default is None.
        """
        # Parent class init creates self.config, self.task_params
        super().__init__(config, load_dir)
        
        self.algo_config = config["algo_config"]
        
        # Create networks
        obs_size = config["observation_size"]
        act_size = config["action_size"]
        hidden_size = self.algo_config["hidden_layer_size"]
        hidden_layer_count = self.algo_config["hidden_layer_count"]
        hidden_layers = [hidden_size] * hidden_layer_count
        
        self.vf = MLP(
            [obs_size] + hidden_layers + [1]
        ).to(DEVICE)
        self.target_vf = MLP(
            [obs_size] + hidden_layers + [1]
        ).to(DEVICE)
        self.qf1 = MLP(
            [obs_size + act_size] + hidden_layers + [1]
        ).to(DEVICE)
        self.qf2 = MLP(
            [obs_size + act_size] + hidden_layers + [1]
        ).to(DEVICE)
        self.policy = GaussianPolicyNet(
            obs_size, act_size, hidden_size, hidden_layer_count,
            log_std_min=self.algo_config["log_std_min"],
            log_std_max=self.algo_config["log_std_max"]
        ).to(DEVICE)
        
        if load_dir is None:
            # Set target value net params to copy of current value net params
            for target_vf_param, vf_param in zip(self.target_vf.parameters(), self.vf.parameters()):
                target_vf_param.data.copy_(vf_param.data)
        else:
            # Load network weights
            self.vf.load_state_dict(torch.load(os.path.join(load_dir, "vf.pt")))
            self.target_vf.load_state_dict(torch.load(os.path.join(load_dir, "target_vf.pt")))
            self.qf1.load_state_dict(torch.load(os.path.join(load_dir, "qf1.pt")))
            self.qf2.load_state_dict(torch.load(os.path.join(load_dir, "qf2.pt")))
            self.policy.load_state_dict(torch.load(os.path.join(load_dir, "policy.pt")))
            
        # Optimizers
        lr = self.algo_config["lr"]
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=lr)
        self.qf1_optimizer = torch.optim.Adam(self.qf1.parameters(), lr=lr)
        self.qf2_optimizer = torch.optim.Adam(self.qf2.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
            
        # Create exp buffer
        self.exp_buffer = ExpBuffer(self.algo_config["exp_buffer_capacity"], obs_size, act_size, load_dir=load_dir)
        
        # Store Policy which uses current network params and automatically accounts for ground-truth offset in returned actions
        self.wrapped_policy = OffsetCorrectedGaussianPolicy(self.policy, self.task_params)
        
        
        
    def save(self, output_dir: str) -> None:
        super().save(output_dir)
        torch.save(self.vf.state_dict(), os.path.join(output_dir, "vf.pt"))
        torch.save(self.target_vf.state_dict(), os.path.join(output_dir, "target_vf.pt"))
        torch.save(self.qf1.state_dict(), os.path.join(output_dir, "qf1.pt"))
        torch.save(self.qf2.state_dict(), os.path.join(output_dir, "qf2.pt"))
        torch.save(self.policy.state_dict(), os.path.join(output_dir, "policy.pt"))
        self.exp_buffer.save(output_dir)
        
    def current_policy(self) -> Policy:
        return self.wrapped_policy
        
    def train_step(self, task_indices: List[int], trajectories: List[List[Trajectory]]) -> dict:
        super().train_step(task_indices, trajectories)
        
        # Add trajectories to buffer AFTER ADDING ACTION OFFSET TO PRODUCE TRUE ENV ACTIONS
        for task_index, task_trajs in zip(task_indices, trajectories):
            for traj in task_trajs:
                traj.actions += self.task_params[task_index] # This is the actual action value that was sent to vanilla LunarLander env
                self.exp_buffer.add_trajectory(traj)
                
                
        # Collect average loss over updates
        q1_losses = []
        q2_losses = []
        vf_losses = []
        policy_losses = []
        
        for _ in range(self.algo_config["updates_per_train_step"]):
            batch = self.exp_buffer.sample(self.algo_config["batch_size"]).to(DEVICE)
            state, action, reward, next_state, done_mask = torch.split(
                batch, [self.config["observation_size"], self.config["action_size"], 1, self.config["observation_size"], 1],
                dim=1
            )
            
            # Q function update - use offline data directly
            q1_pred = self.qf1(state, action)
            q2_pred = self.qf2(state, action)
            q_target = reward + (1 - done_mask) * self.config["discount_rate"] * self.target_vf(next_state).detach()
            q1_loss = F.mse_loss(q1_pred, q_target)
            q2_loss = F.mse_loss(q2_pred, q_target)
            
            self.qf1_optimizer.zero_grad()
            q1_loss.backward()
            self.qf1_optimizer.step()
            
            self.qf2_optimizer.zero_grad()
            q2_loss.backward()
            self.qf2_optimizer.step()
            
            
            
            # Sample new actions and likelihoods from current policy
            new_action, log_prob, eps, mean, log_std = self.policy.evaluate(state)
            new_action_value = torch.min(
                self.qf1(state, new_action),
                self.qf2(state, new_action)
            ).detach()
            
            # V function update
            v_pred = self.vf(state)
            v_target = new_action_value - log_prob.detach()
            vf_loss = F.mse_loss(v_pred, v_target)
            
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()
            
            # Policy update
            policy_loss = (log_prob - new_action_value).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            
            
            # Soft update for target vf params
            tau = self.algo_config["target_vf_update_tau"]
            for target_vf_param, vf_param in zip(self.target_vf.parameters(), self.vf.parameters()):
                target_vf_param.data.copy_(
                    (1 - tau) * target_vf_param.data + tau * vf_param.data
                )
                
                
            # Save loss for logging
            q1_losses.append(q1_loss.item())
            q2_losses.append(q2_loss.item())
            vf_losses.append(vf_loss.item())
            policy_losses.append(policy_loss.item())
            
        # Collect metrics
        metrics = {
            "q1_loss": sum(q1_losses) / len(q1_losses),
            "q2_loss": sum(q2_losses) / len(q2_losses),
            "vf_loss": sum(vf_losses) / len(vf_losses),
            "policy_loss": sum(policy_losses) / len(policy_losses)
        }
        return metrics