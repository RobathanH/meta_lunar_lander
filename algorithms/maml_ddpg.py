import os
import json
import itertools
from typing import Optional, List, Callable, Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from .base import Trainer, Policy, Trajectory
from exp_buffer import MultiTaskExpBuffer
from util import ActionNoise

'''
Uses MAML meta-learning algorithm on top of a DDPG baseline to adapt
to unknown action offsets from experience.
'''


'''
Performs a specified MLP operation on the given parameters manually, allowing grad propagation
'''
class ParamMLP:
    def __init__(self, layer_sizes: List[int], final_activation: Optional[Callable] = None):
        """
        Args:
            layer_sizes (List[int]): Size of each layer, including input and output.
            final_activation (Optional[Callable]): Activation function to call on the output of the network.
        """
        self.layer_sizes = layer_sizes
        self.final_activation = final_activation
        self.params = {}
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            self.params[f"w{i}"] = nn.init.xavier_uniform_(
                torch.empty(
                    out_size,
                    in_size,
                    requires_grad=True,
                    device=DEVICE,
                    dtype=torch.float
                )
            )
            self.params[f"b{i}"] = nn.init.zeros_(
                torch.empty(
                    out_size,
                    requires_grad=True,
                    device=DEVICE,
                    dtype=torch.float
                )
            )
        
    def forward(self, params: Dict[str, torch.Tensor], *x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(x, dim=-1).type(torch.float)
        for i in range(len(self.layer_sizes) - 1):
            #print(x.shape)
            if i > 0:
                x = F.relu(x)
            x = F.linear(
                input=x,
                weight=params[f"w{i}"],
                bias=params[f"b{i}"]
            )
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
        
class MamlPolicy(Policy):
    def __init__(self, actor: ParamMLP, actor_meta_params: Dict[str, torch.Tensor], adapt_func: Callable, exploration_steps: int, action_size: int):
        self.actor = actor
        self.actor_meta_params = actor_meta_params
        self.adapt_func = adapt_func
        self.exploration_steps = exploration_steps
        self.noise = ActionNoise(mu=np.zeros(action_size))
        
        self.reset(0)
        
    def reset(self, task_index: int) -> None:
        # Ignore task_index
        
        # Reset adapted params to None
        self.adapted_actor_params = None
        
        # Reset step counter
        self.step_counter = 0
        
        # Reset mem
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.done_mask = []
        
    def update_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        self.step_counter += 1
        
        if self.adapted_actor_params is None:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.done_mask.append(0)
        
            # Potentially adapt params
            if self.step_counter >= self.exploration_steps:
                _, self.adapted_actor_params = self.adapt_func(
                    train=False,
                    state=torch.from_numpy(np.array(self.states)).type(torch.float).to(DEVICE),
                    action=torch.from_numpy(np.array(self.actions)).type(torch.float).to(DEVICE),
                    reward=torch.from_numpy(np.array(self.rewards).reshape(-1, 1)).type(torch.float).to(DEVICE),
                    next_state=torch.from_numpy(np.array(self.next_states)).type(torch.float).to(DEVICE),
                    done_mask=torch.from_numpy(np.array(self.done_mask).reshape(-1, 1)).type(torch.float).to(DEVICE)
                )
        
    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.from_numpy(state).reshape(1, -1).to(DEVICE)
        if self.adapted_actor_params is None:
            action = self.actor.forward(self.actor_meta_params, state)
        else:
            action = self.actor.forward(self.adapted_actor_params, state)
        action = action.cpu().numpy().reshape(-1)
        return action

class MamlDDPG(Trainer):
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
        
        self.critic = ParamMLP(
            [obs_size + act_size] + hidden_layers + [1]
        )
        self.actor = ParamMLP(
            [obs_size] + hidden_layers + [act_size],
            final_activation=torch.tanh
        )
        
        # Store meta parameters
        self.critic_meta_params = self.critic.params
        self.actor_meta_params = self.actor.params
        self.critic_inner_lr = {
            k: torch.tensor(self.algo_config["inner_lr"], requires_grad=self.algo_config["learn_inner_lr"])
            for k in self.critic_meta_params
        }
        self.actor_inner_lr = {
            k: torch.tensor(self.algo_config["inner_lr"], requires_grad=self.algo_config["learn_inner_lr"])
            for k in self.actor_meta_params
        }
        
        if load_dir is None:
            pass
        else:
            # Load network weights
            stored_critic_meta_params = torch.load(os.path.join(load_dir, "critic.pt"))
            for key in self.critic_meta_params.keys():
                self.critic_meta_params[key].copy_(stored_critic_meta_params[key])
                
            stored_actor_meta_params = torch.load(os.path.join(load_dir, "actor.pt"))
            for key in self.actor_meta_params.keys():
                self.actor_meta_params[key].copy_(stored_actor_meta_params[key])
                
            stored_critic_inner_lr = torch.load(os.path.join(load_dir, "critic_inner_lr.pt"))
            for key in self.critic_inner_lr.keys():
                self.critic_inner_lr[key].copy_(stored_critic_inner_lr[key])
                
            stored_actor_inner_lr = torch.load(os.path.join(load_dir, "actor_inner_lr.pt"))
            for key in self.actor_inner_lr.keys():
                self.actor_inner_lr[key].copy_(stored_actor_inner_lr[key])
                
        # Optimizer
        '''
        self.critic_meta_opt = torch.optim.Adam(itertools.chain(
            self.critic_meta_params.values(),
            self.critic_inner_lr.values()
        ), lr=self.algo_config["outer_lr"])
        self.actor_meta_opt = torch.optim.Adam(itertools.chain(
            self.actor_meta_params.values(),
            self.actor_inner_lr.values()
        ), lr=self.algo_config["outer_lr"])
        '''
        self.meta_opt = torch.optim.Adam(itertools.chain(
            self.critic_meta_params.values(),
            self.actor_meta_params.values(),
            self.critic_inner_lr.values(),
            self.actor_inner_lr.values()
        ), lr=self.algo_config["outer_lr"])
        
        # Create exp buffer for each train task
        self.exp_buffer = MultiTaskExpBuffer(config["num_train_tasks"], self.algo_config["exp_buffer_capacity"], obs_size, act_size, load_dir=load_dir)
        
        # Store Policy which uses current network params and automatically accounts for ground-truth offset in returned actions
        self.wrapped_policy = MamlPolicy(self.actor, self.actor_meta_params, self.adapt, self.algo_config["exploration_steps"], act_size)
        
    def save(self, output_dir: str) -> None:
        super().save(output_dir)
        torch.save(self.critic_meta_params, os.path.join(output_dir, "critic.pt"))
        torch.save(self.actor_meta_params, os.path.join(output_dir, "actor.pt"))
        torch.save(self.critic_inner_lr, os.path.join(output_dir, "critic_inner_lr.pt"))
        torch.save(self.actor_inner_lr, os.path.join(output_dir, "actor_inner_lr.pt"))
        self.exp_buffer.save(output_dir)
        
    def current_policy(self) -> Policy:
        return self.wrapped_policy
    
    def adapt(self, train: bool, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform inner-loop adaptation on an exploration batch.
        """
        critic_params = {
            k: torch.clone(v)
            for k, v in self.critic_meta_params.items()
        }
        actor_params = {
            k: torch.clone(v)
            for k, v in self.actor_meta_params.items()
        }
        
        for i in range(self.algo_config["num_inner_updates"]):
            
            # Update critic
            next_action = self.actor.forward(actor_params, next_state)
            q_target = reward + (1 - done_mask) * self.config["discount_rate"] * self.critic.forward(critic_params, next_state, next_action)
            q_pred = self.critic.forward(critic_params, state, action)
            critic_loss = F.mse_loss(q_pred, q_target)
            
            grads = {
                k: torch.autograd.grad(critic_loss, p, create_graph=train, retain_graph=True)[0]
                for k, p in critic_params.items()
            }
            critic_params = {
                k: p - self.critic_inner_lr[k] * grads[k]
                for k, p in critic_params.items()
            }
            
            # Update actor
            new_action = self.actor.forward(actor_params, state)
            actor_loss = -self.critic.forward(critic_params, state, new_action).mean()
            
            grads = {
                k: torch.autograd.grad(actor_loss, p, create_graph=train, retain_graph=True)[0]
                for k, p in actor_params.items()
            }
            actor_params = {
                k: p - self.actor_inner_lr[k] * grads[k]
                for k, p in actor_params.items()
            }
            
        return critic_params, actor_params
    
    def train_step(self, task_indices: List[int], trajectories: List[List[Trajectory]]) -> dict:
        super().train_step(task_indices, trajectories)
        
        # Add trajectories to buffer
        for task_index, task_trajs in zip(task_indices, trajectories):
            for traj in task_trajs:
                self.exp_buffer.add_trajectory(task_index, traj)
                
        # Save average losses
        critic_losses = []
        actor_losses = []
        
        explore_steps = self.algo_config["exploration_steps"] 
        for _ in range(self.algo_config["updates_per_train_step"]):
                
            # Get exploration batch from initial moves in each trajectory
            for task_index, task_trajs in zip(task_indices, trajectories):
                for traj in task_trajs:
                    if len(traj) < explore_steps:
                        continue
                    
                    # Inner-loop adaptation
                    adapted_critic_params, adapted_actor_params = self.adapt(
                        train=True,
                        state=torch.from_numpy(traj.states[:explore_steps]).type(torch.float).to(DEVICE),
                        action=torch.from_numpy(traj.actions[:explore_steps]).type(torch.float).to(DEVICE),
                        reward=torch.from_numpy(traj.rewards[:explore_steps]).type(torch.float).to(DEVICE),
                        next_state=torch.from_numpy(traj.next_states[:explore_steps]).type(torch.float).to(DEVICE),
                        done_mask=torch.from_numpy(traj.done_mask[:explore_steps]).type(torch.float).to(DEVICE)
                    )
                    
                    # Outer-loop update
                    batch = self.exp_buffer.sample(task_index, self.algo_config["batch_size"]).to(DEVICE)
                    state, action, reward, next_state, done_mask = torch.split(
                        batch, [self.config["observation_size"], self.config["action_size"], 1, self.config["observation_size"], 1],
                        dim=1
                    )
                    
                    next_action = self.actor.forward(adapted_actor_params, next_state)
                    q_target = reward + (1 - done_mask) * self.config["discount_rate"] * self.critic.forward(adapted_critic_params, next_state, next_action)
                    q_pred = self.critic.forward(adapted_critic_params, state, action)
                    critic_loss = F.mse_loss(q_pred, q_target)
                    '''
                    self.critic_meta_opt.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    self.critic_meta_opt.step()
                    '''
                    
                    new_action = self.actor.forward(adapted_actor_params, state)
                    actor_loss = -self.critic.forward(adapted_critic_params, state, new_action).mean()
                    '''
                    self.actor_meta_opt.zero_grad()
                    actor_loss.backward()
                    self.actor_meta_opt.step()
                    '''
                    
                    total_loss = critic_loss + actor_loss
                    self.meta_opt.zero_grad()
                    total_loss.backward()
                    self.meta_opt.step()
                    
                    # Save loss for logging
                    critic_losses.append(critic_loss.item())
                    actor_losses.append(actor_loss.item())
                    
        # Collect metrics
        metrics = {}
        metrics["critic_loss"] = sum(critic_losses) / len(critic_losses)
        metrics["actor_loss"] = sum(actor_losses) / len(actor_losses)
        return metrics
    
    
    
    
    
    
    
