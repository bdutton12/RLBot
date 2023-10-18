import numpy as np
import torch

from actor import ActorNN
from critic import CriticNN
from memory import PPOMemory

"""
Class representing our agent in the env 
"""
class Agent:
  """
  Constructor
 
  """
  def __init__(self, n_actions, input_dims, gamma=0.99, eta=0.0003, gae_lambda=0.95,\
                policy_clip=0.2, batch_size=64, n_epochs=10) -> None:
    self.gamma = gamma
    self.policy_clip = policy_clip
    self.n_epochs = n_epochs
    self.gae_lambda = gae_lambda

    self.actor = ActorNN(n_actions, input_dims, eta)
    self.critic = CriticNN(input_dims, eta)
    self.memory = PPOMemory(batch_size)
  
  """
  Save data to memory
 
  """
  def store_to_memory(self, state, action, probs, vals, reward, done) -> None:
    self.memory.save_to_memory(state, action, probs, vals, reward, done)

  """
  Save models to files 
  """
  def save_models(self) -> None:
    print('[SAVING] saving models...')
    self.actor.save_network()
    self.critic.save_network()
    print('Models saved.')

  """
  Load models from file 
  """
  def load_models(self):
    print('[LOADING] loading models...')
    self.actor.load_network()
    self.critic.load_network()
    print('Models loaded.')

  """
  Choose an action based on the env observation 
  """
  def choose_action(self, observation):
    state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

    distribution = self.actor(state)
    value = self.critic(state)
    action = distribution.sample()

    probs = torch.squeeze(distribution.log_prob(action)).item()
    action = torch.squeeze(action).item()
    value = torch.squeeze(value).item()

    return action, probs, value
  
  """
  Train the agent 
  """
  def learn(self):
    # Train model with n_epochs for each batch
    for _ in range(self.n_epochs):
      # Get random samples from memory
      state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

      # Initialze critic values and advantage array
      values = vals_arr
      advantage = np.zeros(len(reward_arr), dtype=np.float32)

      for t in range(len(reward_arr) - 1):
        discount = 1
        a_t = 0
        # Compute advantage at each time step, tells us benifit of new state over the old state
        for k in range(t, len(reward_arr) - 1):
          a_t += discount*(reward_arr[k] + self.gamma+values[k+1]*(1-int(dones_arr[k])) - values[k])
          discount += self.gamma*self.gae_lambda
        
        advantage[t] = a_t
      advantage = torch.tensor(advantage).to(self.actor.device)

      values = torch.tensor(values).to(self.actor.device)
      # Loop through all batches
      for batch in batches:
        # Get associated data for each batch
        states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
        old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
        actions = torch.tensor(action_arr[batch]).to(self.actor.device)

        distribution = self.actor(states)
        critic_value = self.critic(states)

        critic_value = torch.squeeze(critic_value)

        new_probs = distribution.log_prob(actions)
        prob_ratio = new_probs.exp() / old_probs.exp()

        # Calculate weighted probs, prob ratio * advantage
        # and clipped probs, clamps the prob ratio * advantage to between 1-self.policy_clip and 1+self.policy_clip 
        weighted_probs = advantage[batch]*prob_ratio
        weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]

        # Find the actor loss, 
        # take the minimum of weighted and clipped probs to ensure model is not updating too much at one time
        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

        # Calculate critic loss
        # MSE of returns (advantage + critic_values) minus the critic values
        returns = advantage[batch] + values[batch]
        critic_loss = (returns - critic_value)**2
        critic_loss = critic_loss.mean()

        # Total loss is actor_loss + c*critic_los
        # where c is some constant
        c = 0.5
        total_loss = actor_loss + c*critic_loss

        # Zero the gradients of the optimzers and backward propogate the loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
    
    self.memory.clear_memory()