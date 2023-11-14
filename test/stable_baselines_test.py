import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Get the current reward
        current_reward = self.model.ep_info_buffer
        print(current_reward)
        
        # You can log or store the reward as needed
        print(f"Current Reward: {current_reward}")

        return True

# Parallel environments
env = gym.make("CartPole-v1")
env = Monitor(env, 'data')

tmp_path = "data"
csv_logger = configure(tmp_path, ["stdout", "csv"])

# callback = RewardCallback()

model = PPO("MlpPolicy", env, verbose=1)
model.set_logger(csv_logger)
model.learn(total_timesteps=25000)

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)