import os
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

"""
Class representing the actor network in our PPO model
"""
class ActorNN(nn.Module):
  """
  Constructor

  """
  def __init__(self, n_actions, input_dims, eta, fc1_dims=256, fc2_dims=256, file_save='saved_models') -> None:
    super(ActorNN, self).__init__()

    self.file_save = os.path.join(file_save, 'actor_ppo')
    self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
    )

    self.optimizer = optim.Adam(self.parameters(), lr=eta)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  """
  Get probabilistic distribution of actions to take based on an env state

  @param state :: The env state to input
  @return The distribution representing probability of each action to take
  """
  def forward(self, state) -> Any:
    distribution = self.actor(state)
    distribution = Categorical(distribution)

    return distribution
  
  """
  Save the actor network to a file
  """
  def save_network(self) -> None:
    torch.save(self.state_dict(), self.file_save)

  """
  Load the actor network from a file
  """
  def load_network(self) -> None:
    self.load_state_dict(torch.load(self.file_save))