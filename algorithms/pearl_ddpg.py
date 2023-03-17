import os
import json
import itertools
import heapq
from typing import Optional, List, Callable, Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from .base import Trainer, Policy, Trajectory
from networks import MLP
from exp_buffer import MultiTaskExpBuffer
from util import ActionNoise

'''
Uses PEARL meta-learning algorithm on top of DDPG baseline
'''

class PearlDDPGPolicy(Policy):
    def __init__(self, actor_net: MLP, sample_latent_func: Callable, update_latent_every_action: bool, exploration_steps: int, state_size: int, action_size: int, latent_size: int):
        self.actor_net = actor_net
        self.sample_latent_func = sample_latent_func
        self.update_latent_every_action = update_latent_every_action
        self.exploration_steps = exploration_steps
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        
    def reset(self, action_offset: np.ndarray, eval: bool = False) -> None:
        # Ignore task params
        
        # Set eval flag
        self.eval = eval
        
        # Reset action noise
        self.noise = ActionNoise(mu=np.zeros(self.action_size))
        
        # Sample latent from uniform prior
        self.latent = torch.normal(torch.zeros(1, self.latent_size), torch.ones(1, self.latent_size)).to(DEVICE)
        
        # Maintain latent mean/std after computation
        self.latent_mean = None
        self.latent_var = None
        
        # Reset step counter
        self.step_counter = 0
        
        # Reset mem
        self.mem = []
        
    @torch.no_grad()
    def update_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        self.step_counter += 1
        
        new_context_element = torch.cat([
            torch.from_numpy(state).type(torch.float),
            torch.from_numpy(action).type(torch.float),
            torch.tensor([reward]).type(torch.float),
            torch.from_numpy(next_state).type(torch.float)
        ], dim=0).reshape(1, 1, -1)
        
        if self.update_latent_every_action:
            # Sample latent distribution factor for latest transition seen, then multiply it into the running latent distribution and resample latent
            new_latent_mean_element, new_latent_var_element, _ = self.sample_latent_func(new_context_element.to(DEVICE))
            if self.step_counter == 1:
                self.latent_mean = new_latent_mean_element
                self.latent_var = new_latent_var_element
            else:
                latent_means = torch.cat([self.latent_mean.unsqueeze(1), new_latent_mean_element.unsqueeze(1)], dim=1)
                latent_vars = torch.cat([self.latent_var.unsqueeze(1), new_latent_var_element.unsqueeze(1)], dim=1)

                self.latent_var = torch.reciprocal(torch.reciprocal(latent_vars).sum(dim=1))
                self.latent_mean = self.latent_var * (latent_means / latent_vars).sum(dim=1)
                
            # Resample latent
            self.latent = torch.normal(self.latent_mean, self.latent_var.sqrt())
        else:
            if self.step_counter <= self.exploration_steps:
                self.mem.append(new_context_element)
                
                if self.step_counter == self.exploration_steps:
                    context_batch = torch.cat(self.mem, dim=1).to(DEVICE)
                    self.latent_mean, self.latent_var, self.latent = self.sample_latent_func(context_batch)
        
    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.from_numpy(state).type(torch.float).unsqueeze(0).to(DEVICE)
        action = self.actor_net(state, self.latent)
        action = action.cpu().numpy().flatten()
        
        if not self.eval:
            action += self.noise()

        return action
                
        

class PearlDDPG(Trainer):
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
        latent_size = self.algo_config["task_encoding_size"]
        hidden_size = self.algo_config["hidden_layer_size"]
        hidden_layer_count = self.algo_config["hidden_layer_count"]
        hidden_layers = [hidden_size] * hidden_layer_count
        
        self.critic = MLP(
            [obs_size + act_size + latent_size] + hidden_layers + [1]
        ).to(DEVICE)
        self.target_critic = MLP(
            [obs_size + act_size + latent_size] + hidden_layers + [1]
        ).to(DEVICE)
        
        self.actor = MLP(
            [obs_size + latent_size] + hidden_layers + [act_size],
            final_activation=torch.tanh
        ).to(DEVICE)
        self.target_actor = MLP(
            [obs_size + latent_size] + hidden_layers + [act_size],
            final_activation=torch.tanh
        ).to(DEVICE)
        
        self.task_encoder = MLP(
            [obs_size + act_size + 1 + obs_size] + hidden_layers + [2 * latent_size]
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
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.algo_config.get("critic_lr", self.algo_config["lr"]) or self.algo_config["lr"])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.algo_config.get("actor_lr", self.algo_config["lr"]) or self.algo_config["lr"])
        self.encoder_opt = torch.optim.Adam(self.task_encoder.parameters(), lr=self.algo_config.get("encoder_lr", self.algo_config["lr"]) or self.algo_config["lr"])
        
        self.exp_buffer = MultiTaskExpBuffer(config["num_train_tasks"], self.algo_config["exp_buffer_capacity"], obs_size, act_size, load_dir=load_dir)
        
        # Task encoder exp buffer doesn't get saved or loaded, since it should focus primarily on online data
        self.task_encoder_exp_buffer = MultiTaskExpBuffer(config["num_train_tasks"], self.algo_config["task_encoder_exp_buffer_capacity"], obs_size, act_size)
        
        self.wrapped_policy = PearlDDPGPolicy(self.actor, self.sample_latent, self.algo_config["update_latent_every_action"],
                                              self.algo_config["exploration_steps"], obs_size, act_size, latent_size)
        
        if self.algo_config["prioritized_level_replay"]:
            self.max_return = np.full(config["num_train_tasks"], np.nan)
            self.regret = np.full(config["num_train_tasks"], np.nan)
            self.prioritized_task_indices = set() # Set of task indices in plr buffer
        
    def save(self, output_dir: str) -> None:
        super().save(output_dir)
        torch.save(self.critic.state_dict(), os.path.join(output_dir, "critic.pt"))
        torch.save(self.target_critic.state_dict(), os.path.join(output_dir, "target_critic.pt"))
        torch.save(self.actor.state_dict(), os.path.join(output_dir, "actor.pt"))
        torch.save(self.target_actor.state_dict(), os.path.join(output_dir, "target_actor.pt"))
        torch.save(self.task_encoder.state_dict(), os.path.join(output_dir, "task_encoder.pt"))
        self.exp_buffer.save(output_dir)
        
    def current_policy(self) -> Policy:
        return self.wrapped_policy
    
    def sample_latent(self, context_batch: torch.Tensor):
        """
        Args:
            context_batch (torch.Tensor): shape = (batch, exploration_steps, state + action + reward + next_state concat)

        Returns:
            torch.Tensor: Mean: shape = (batch, latent_size)
            torch.Tensor: Var: shape = (batch, latent_size)
            torch.Tensor: Sample: shape = (batch, latent_size)
        """
        episode_count, step_count, context_size = context_batch.shape
        latent_size = self.algo_config["task_encoding_size"]
        
        # Compute task encodings dist for each individual context tuple
        task_encoder_output = self.task_encoder(context_batch.reshape(-1, context_size)).reshape(episode_count, step_count, 2 * latent_size)
        latent_mean_per_tuple = task_encoder_output[..., :latent_size]
        latent_var_per_tuple = torch.clamp(F.softplus(task_encoder_output[..., latent_size:]), min=1e-7)
        
        # Compute product of dists over all context tuples per episode
        latent_var = torch.reciprocal(torch.reciprocal(latent_var_per_tuple).sum(dim=1))
        latent_mean = latent_var * (latent_mean_per_tuple / latent_var_per_tuple).sum(dim=1)
        latent_dist = torch.distributions.Normal(latent_mean, latent_var.sqrt())
        
        # Sample task encoding
        latent = latent_dist.rsample() # shape = (episode_count, latent_size)
        
        return latent_mean, latent_var, latent
    
    def train_step(self, task_indices: List[int], trajectories: List[List[Trajectory]]) -> dict:
        super().train_step(task_indices, trajectories)
        
        # Add trajectories to buffer
        for task_index, task_trajs in zip(task_indices, trajectories):
            for traj in task_trajs:
                self.exp_buffer.add_trajectory(task_index, traj)
                if self.algo_config["update_latent_every_action"]:
                    self.task_encoder_exp_buffer.add_trajectory(task_index, traj)
                else:
                    self.task_encoder_exp_buffer.add_trajectory(task_index, traj, max_length=self.algo_config["exploration_steps"])
                    
        # Train on all tasks with new data
        task_batch_indices = task_indices
                
        # Save average losses
        critic_losses = []
        actor_losses = []
        kl_losses = []
        mean_latent_stds = []
        latent_mean_task_distances = []
        
        # Useful constants
        explore_steps = self.algo_config["exploration_steps"]
        latent_size = self.algo_config["task_encoding_size"]
        
        for _ in range(self.algo_config["updates_per_train_step"]):
            # Sample context for each episode
            context = torch.cat([
                self.task_encoder_exp_buffer.sample(context_task_index, explore_steps)[None, :, :-1] # Drop done mask
                for context_task_index in task_batch_indices
            ], dim=0).to(DEVICE)
            
            # Sample off-policy data for task-conditioned policy update
            batch = torch.cat([
                self.exp_buffer.sample(task_index, self.algo_config["batch_size"]).unsqueeze(0).to(DEVICE)
                for task_index in task_batch_indices
            ], dim=0) # shape = (task_batch_size, sample_batch_size, all_features_concatenated_size)
            batch = batch.flatten(end_dim=1) # shape = (task_batch_size * sample_batch_size, all_features_concat_size)
            state, action, reward, next_state, done_mask = torch.split(
                batch, [self.config["observation_size"], self.config["action_size"], 1, self.config["observation_size"], 1],
                dim=1
            )
            
            # Compute task encoding
            latent_mean, latent_var, latent = self.sample_latent(context)
            
            # Expand task encoding so it will match up with policy batch data
            # shape = (task_batch_size * sample_batch_size, latent_size)
            latent = latent.unsqueeze(1).expand(-1, self.algo_config["batch_size"], -1).reshape(-1, latent_size)
            
            # critic loss (q_pred may propagate back to task encoder)
            with torch.no_grad():
                next_action = self.target_actor(next_state, latent)
                q_target = reward + (1 - done_mask) * self.config["discount_rate"] * self.target_critic(next_state, next_action, latent)
            if self.algo_config["task_encoder_uses_critic_loss"]:
                critic_latent = latent
            else:
                critic_latent = latent.detach()        
            q_pred = self.critic(state, action, critic_latent)
            critic_loss = F.mse_loss(q_pred, q_target)
            
            # actor loss (does NOT propagate to task encoder)
            if self.algo_config["task_encoder_uses_actor_loss"]:
                actor_latent = latent
            else:
                actor_latent = latent.detach()
            new_action = self.actor(state, actor_latent)
            actor_loss = -self.critic(state, new_action, latent.detach()).mean()
            
            # task encoder kl-div loss
            kl_loss = self.algo_config["kl_lambda"] * 0.5 * (-torch.log(latent_var) - 1 + torch.square(latent_mean) + latent_var).sum(dim=-1).mean()
            
            # Update step
            self.encoder_opt.zero_grad()
            self.critic_opt.zero_grad()
            self.actor_opt.zero_grad()
            kl_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=self.algo_config["task_encoder_uses_critic_loss"] and self.algo_config["task_encoder_uses_actor_loss"])
            actor_loss.backward()
            self.encoder_opt.step()
            self.critic_opt.step()
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
            kl_losses.append(kl_loss.item())
            mean_latent_stds.append(latent_var.sqrt().mean())
            
            # dist between each latent-mean (one for each task collected)
            latent_mean = latent_mean.detach().cpu()
            latent_mean_task_dmatrix = torch.cdist(latent_mean, latent_mean)
            task_batch_size = len(latent_mean)
            latent_mean_task_distance = torch.sum(latent_mean_task_dmatrix * (1 - torch.eye(task_batch_size))) / (task_batch_size * (task_batch_size - 1))
            latent_mean_task_distances.append(latent_mean_task_distance)
            
        
        # Update prioritized replay buffer
        if self.algo_config["prioritized_level_replay"]:
            self.update_regret_estimates(task_indices, trajectories)
            
            # Update prioritized task buffer
            unprioritized_task_indices = [task_index for task_index in task_indices if task_index not in self.prioritized_task_indices]
            for task_index in unprioritized_task_indices:
                if len(self.prioritized_task_indices) < self.algo_config["plr_buffer_size"]:
                    self.prioritized_task_indices.add(task_index)
                else:
                    worst_prioritized_task = min(self.prioritized_task_indices, key=lambda i: self.regret[i])
                    if self.regret[task_index] > self.regret[worst_prioritized_task]:
                        self.prioritized_task_indices.remove(worst_prioritized_task)
                        self.prioritized_task_indices.add(task_index)
                
                
            
        # Collect metrics
        metrics = {}
        metrics["critic_loss"] = sum(critic_losses) / len(critic_losses)
        metrics["actor_loss"] = sum(actor_losses) / len(actor_losses)
        metrics["kl_loss"] = sum(kl_losses) / len(kl_losses)
        metrics["latent_std"] = sum(mean_latent_stds) / len(mean_latent_stds)
        metrics["latent_mean_task_distance"] = sum(latent_mean_task_distances) / len(latent_mean_task_distances)
        return metrics
    
    
    
    
    '''
    Helper functions for prioritized level replay
    '''
    
    def choose_next_train_task_indices(self) -> Optional[List[int]]:
        """Called before collecting rollouts, allows Trainer to actively select the next tasks to collect rollouts on.
        By default, returns a random selection of training task indices.
        """
        if not self.algo_config["prioritized_level_replay"]:
            return super().choose_next_train_task_indices()
        
        # Use random sampling until buffer is full
        if len(self.prioritized_task_indices) < self.algo_config["plr_buffer_size"]:
            return super().choose_next_train_task_indices()

        task_indices = []
        for i in range(self.config["train_task_batch_size"]):
            if np.random.rand() <= self.algo_config["plr_probability"]:
                task_indices.append(np.random.choice(list(self.prioritized_task_indices)))
            else:
                unprioritized_task_indices = list(set(range(self.config["num_train_tasks"])) - self.prioritized_task_indices)
                task_indices.append(np.random.choice(unprioritized_task_indices))
        return task_indices
        
        
    @torch.no_grad()        
    def update_regret_estimates(self, task_indices: List[int], trajectories: List[List[Trajectory]]):
        for task_index, task_trajs in zip(task_indices, trajectories):
            # Update max return for this task
            for traj in task_trajs:
                rewards = traj.rewards
                discounted_return = np.sum(rewards * np.power(self.config["discount_rate"], np.arange(len(rewards))))
                if np.isnan(self.max_return[task_index]) or discounted_return > self.max_return[task_index]:
                    self.max_return[task_index] = discounted_return
            
            # Estimate average value for task
            value = []
            for traj in task_trajs:
                # Compute latent mean (no sampling)
                context_steps = self.algo_config["exploration_steps"]
                context = torch.cat([
                    torch.from_numpy(traj.states[:context_steps]).type(torch.float),
                    torch.from_numpy(traj.actions[:context_steps]).type(torch.float),
                    torch.from_numpy(traj.rewards[:context_steps]).type(torch.float),
                    torch.from_numpy(traj.next_states[:context_steps]).type(torch.float)
                ], dim=1)[None, :, :].to(DEVICE)
                latent_mean, _, _ = self.sample_latent(context)
                
                # Compute estimated value at each step
                states = torch.from_numpy(traj.states).type(torch.float).to(DEVICE)
                latent = latent_mean.expand(len(states), -1)
                
                chosen_actions = self.actor(states, latent)
                value.append(self.critic(states, chosen_actions, latent).cpu().mean())
            value = np.mean(value)

            # Update regret estimates
            self.regret[task_index] = self.max_return[task_index] - value