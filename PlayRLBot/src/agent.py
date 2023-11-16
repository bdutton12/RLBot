# import os
from stable_baselines3.ppo import PPO


class Agent:
    def __init__(self):
        learner = PPO(policy="MlpPolicy", verbose=1)
        self.agent = learner.load("../saved_model/PPO_model.zip")

    def act(self, state):
        # Evaluate your model here
        return self.agent.predict(state)
