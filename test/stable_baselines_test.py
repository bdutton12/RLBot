import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

tmp_path = "data"
csv_logger = configure(tmp_path, ["stdout", "csv"])

model = PPO("MlpPolicy", vec_env, verbose=1)
model.set_logger(csv_logger)
model.learn(total_timesteps=25000)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)