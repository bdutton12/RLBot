import numpy as np
from typing import Tuple


"""
Class representing the memory of our PPO model
"""
class PPOMemory:
  """
  Constructor

  @param batch_size :: the size of each batch taken from memory
  """
  def __init__(self, batch_size: int) -> None:
    self.states = []
    self.probs = []
    self.vals = []
    self.actions = []
    self.rewards = []
    self.dones = []

    self.batch_size = batch_size

  """
  Generates batches from model memory

  @returns Tuple of numpy arrays (states, actions, probs, vals, rewards, dones, batches)
  """
  def generate_batches(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_states = len(self.states)
    batch_start = np.arange(0, n_states, self.batch_size)
    indices = np.arange(n_states, dtype=np.int64)
    np.random.shuffle(indices)
    batches = [indices[i:i+self.batch_size] for i in batch_start]

    return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals),\
      np.array(self.rewards), np.array(self.dones), batches
  
  """
  Save items to model memory (state, action, probs, vals, reward, done)

  @param state :: The env state
  @param action :: The action taken
  @param probs :: The probabilities to take each action
  @param vals :: TODO
  @param reward :: The reward given for the action
  @param done :: The env done flag
  """
  def save_to_memory(self, state, action, probs, vals, reward, done) -> None:
    self.states.append(state)
    self.actions.append(action)
    self.probs.append(probs)
    self.vals.append(vals)
    self.rewards.append(reward)
    self.dones.append(done)

  """
  Clear the model memory
  """
  def clear_memory(self) -> None:
    self.states = []
    self.probs = []
    self.actions = []
    self.rewards = []
    self.dones = []
    self.vals = []
