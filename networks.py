from typing import Optional, List, Tuple, Callable
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


'''
Simple MLP with ReLU activation. Automatically concatenates inputs if multiple are provided.
'''
class MLP(nn.Module):
    def __init__(self, layer_sizes: List[int], final_activation: Optional[Callable] = None):
        """
        Args:
            layer_sizes (List[int]): Size of each layer, including input and output.
            final_activation (Optional[Callable]): Activation function to call on the output of the network.
        """
        super().__init__()
        
        self.final_activation = final_activation
        
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Add relu between layers
            if len(layers):
                layers.append(nn.ReLU())
            layers.append(nn.Linear(in_size, out_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat(x, dim=1))
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out    
    
'''
Gaussian policy network.
Assumes output action should be transformed to (-1, 1) range
'''
class GaussianPolicyNet(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hidden_size: int, hidden_layer_count: int,
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = MLP([obs_size] + [hidden_size] * hidden_layer_count + [2 * act_size])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_std = torch.split(self.net(x), 2, dim=1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collect action likelihood info needed for policy update

        Args:
            state (torch.Tensor): Batch of states. Shape = (batch_size, obs_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        mean, log_std = self(state)
        std = log_std.exp()
        
        z = Normal(torch.zeros_like(mean), torch.ones_like(std)).sample().to(mean.device)
        raw_action = mean + std * z # before being scaled to (-1, 1) range
        action = torch.tanh(raw_action)
        log_prob = (Normal(mean, std).log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=1, keepdims=True) # (IDK what 2nd term does)
        return action, log_prob, z, mean, log_std
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select an action for the given state.

        Args:
            state (torch.Tensor): Single state. Shape = (obs_size,)

        Returns:
            torch.Tensor: action. Shape = (act_size,)
        """
        state = state.reshape(1, -1)
        mean, log_std = self(state)
        std = log_std.exp()
        
        z = Normal(torch.zeros_like(mean), torch.ones_like(std)).sample().to(mean.device)
        action = torch.tanh(mean + std * z)
        return action[0]