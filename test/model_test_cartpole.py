import gym
import numpy as np
import sys

sys.path.append('/Users/brucesmith/MSU/CSE404/TeamProject/RLBot/model')
from model.agent import Agent
from model.utils import plot_learning_curve

if __name__ == '__main__':
  figure_file = '/Users/brucesmith/MSU/CSE404/TeamProject/RLBot/model/saved_models/results.png'
  env = gym.make('CartPole-v1')
  N = 20
  batch_size = 5
  n_epochs = 4
  eta = 0.0003
  policy_clip = 0.2
  agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                  eta=eta, n_epochs=n_epochs, 
                  input_dims=env.observation_space.shape, policy_clip=policy_clip)
  
  n_games = 300

  best_score = env.reward_range[0]
  score_history = []

  learn_iters = 0
  avg_score = 0
  n_steps = 0

  for i in range(n_games):
    observation, info = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        print(action, prob, val)
        observation_, reward, done, trunc, info = env.step(action)
        n_steps += 1
        score += reward
        agent.store_to_memory(observation, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
  x = [i+1 for i in range(len(score_history))]
  plot_learning_curve(x, score_history, figure_file)