# import os
from stable_baselines3.ppo import PPO


class Agent:
    def __init__(self):
        self.agent = PPO.load("../saved_model/PPO_model.zip", env=None)

    def act(self, state):
        # Evaluate your model here
        return self.agent.predict(state)[0]
