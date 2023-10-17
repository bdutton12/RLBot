import os
import torch
import torch.nn as nn
import torch.optim as optim

class CriticNN(nn.Module):
  """
  Constructor

  """
  def __init__(self, input_dims, eta, fc1_dims=256, fc2_dims=256, file_save='saved_models') -> None:
    super(CriticNN, self).__init__()

    self.file_save = os.path.join(file_save, 'critic_ppo')
    self.critic = nn.Sequential(
      nn.Linear(*input_dims, fc1_dims),
      nn.ReLU(),
      nn.Linear(fc1_dims, fc2_dims),
      nn.ReLU(),
      nn.Linear(fc2_dims, 1)
    )

    self.optimizer = optim.Adam(self.parameters(), lr=eta)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  """
  Get value outputed from critic network from input state

  @param stte :: The env state to input
  @return The output from the critic network 
  """
  def forward(self, state):
    value = self.critic(state)

    return value
  
  """
  Save the critic network to a file
  """
  def save_network(self) -> None:
    torch.save(self.state_dict(), self.file_save)

  """
  Load the critic network from a file
  """
  def load_network(self) -> None:
    self.load_state_dict(torch.load(self.file_save))