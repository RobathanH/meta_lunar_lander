import os
import json
from typing import Optional, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from .base import Trainer, Policy, Trajectory
from networks import MLP
from exp_buffer import ExpBuffer
from util import ActionNoise


'''
DDPG vanilla-RL algorithm, ignores the offset and treats the meta-RL
problem like a regular RL problem.
'''



class DDPGPolicy(Policy):
    def __init__(self, actor_net: MLP, action_size: int):
        self.actor_net = actor_net
        self.action_size = action_size
        
    def reset(self, action_offset: np.ndarray, eval: bool = False) -> None:
        # Ignore task_index
        
        # Set eval flag
        self.eval = eval
        
        # Reset action noise
        self.noise = ActionNoise(mu=np.zeros(self.action_size))
        
    def update_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        pass
        
    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.from_numpy(state).to(DEVICE).reshape(1, -1)
        action = self.actor_net(state).cpu().numpy().reshape(-1)
        
        if not self.eval:
            action += self.noise()
            
        return action




class NaiveDDPG(Trainer):
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
        
        self.critic = MLP(
            [obs_size + act_size] + hidden_layers + [1]
        ).to(DEVICE)
        self.target_critic = MLP(
            [obs_size + act_size] + hidden_layers + [1]
        ).to(DEVICE)
        
        self.actor = MLP(
            [obs_size] + hidden_layers + [act_size],
            final_activation=torch.tanh
        ).to(DEVICE)
        self.target_actor = MLP(
            [obs_size] + hidden_layers + [act_size],
            final_activation=torch.tanh
        ).to(DEVICE)
        
        if load_dir is None:
            # Copy target critic params from current critic params
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)
        else:
            # Load network weights
            self.critic.load_state_dict(torch.load(os.path.join(load_dir, "critic.pt")))
            self.target_critic.load_state_dict(torch.load(os.path.join(load_dir, "target_critic.pt")))
            self.actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pt")))
            self.target_actor.load_state_dict(torch.load(os.path.join(load_dir, "target_actor.pt")))
            
        # Optimizers
        lr = self.algo_config["lr"]
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Create exp buffer
        self.exp_buffer = ExpBuffer(self.algo_config["exp_buffer_capacity"], obs_size, act_size, load_dir=load_dir)
        
        # Store Policy which uses current network params
        self.wrapped_policy = DDPGPolicy(self.actor, act_size)
        
        
        
    def save(self, output_dir: str) -> None:
        super().save(output_dir)
        torch.save(self.critic.state_dict(), os.path.join(output_dir, "critic.pt"))
        torch.save(self.target_critic.state_dict(), os.path.join(output_dir, "target_critic.pt"))
        torch.save(self.actor.state_dict(), os.path.join(output_dir, "actor.pt"))
        torch.save(self.target_actor.state_dict(), os.path.join(output_dir, "target_actor.pt"))
        self.exp_buffer.save(output_dir)
        
    def current_policy(self) -> Policy:
        return self.wrapped_policy
    
    def train_step(self, task_indices: List[int], trajectories: List[List[Trajectory]]) -> dict:
        super().train_step(task_indices, trajectories)
        
        # Add trajectories to buffer
        for task_trajs in trajectories:
            for traj in task_trajs:
                self.exp_buffer.add_trajectory(traj)
                
        # Save average losses
        critic_losses = []
        actor_losses = []
                
        for _ in range(self.algo_config["updates_per_train_step"]):
            batch = self.exp_buffer.sample(self.algo_config["batch_size"]).to(DEVICE)
            state, action, reward, next_state, done_mask = torch.split(
                batch, [self.config["observation_size"], self.config["action_size"], 1, self.config["observation_size"], 1],
                dim=1
            )
            
            # Update critic
            next_action = self.target_actor(next_state).detach()
            q_target = reward + (1 - done_mask) * self.config["discount_rate"] * self.target_critic(next_state, next_action).detach()
            critic_loss = F.mse_loss(self.critic(state, action), q_target)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            
            # Update actor
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            # Soft update for target networks
            tau = self.algo_config["target_net_update_tau"]
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    (1 - tau) * target_param.data + tau * param.data
                )
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    (1 - tau) * target_param.data + tau * param.data
                )
            
            # Save loss for logging
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            
        # Collect metrics
        metrics = {}
        metrics["critic_loss"] = sum(critic_losses) / len(critic_losses)
        metrics["actor_loss"] = sum(actor_losses) / len(actor_losses)
        return metrics
    
    
    
    