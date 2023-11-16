import numpy as np
import rlgym
from rlgym.utils.reward_functions.common_rewards import VelocityReward
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions.common_conditions import (
    TimeoutCondition,
    GoalScoredCondition,
)
from reward_functions import AlignAndDistanceReward, HybridReward
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

from stable_baselines3.ppo import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor


"""
Observation Object for RLGym

Gets current state of ball and all players and inverts the values if bot is on team Orange
so that the model doesn't need to worry about team
"""


class UnbiasedObservationBuilder(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ):
        obs = []

        # If this observation is being built for a player on the orange team, we need to invert all the physics data we use.
        inverted = player.team_num == common_values.ORANGE_TEAM

        if inverted:
            obs += state.inverted_ball.serialize()
        else:
            obs += state.ball.serialize()

        for player in state.players:
            if inverted:
                obs += player.inverted_car_data.serialize()
            else:
                obs += player.car_data.serialize()

        return np.asarray(obs, dtype=np.float32)


"""
Main Entry Point for Training

Defines an environment in RLGym, then passes it to an SB3 wrapper that initializes 
"""







if __name__ == "__main__":
    # specify model

    reward_model = '1'
    reward_model = '2'

    if reward_model == '1':
        reward_fn = AlignAndDistanceReward()
        log_path = "data"
        saved_model_dir = "./saved_model/PPO_model.zip"
    elif reward_model == '2':
        reward_fn = HybridReward()
        log_path = "data2"
        saved_model_dir = "./saved_model/PPO_model2.zip"


    # Make RLGym environment
    default_tick_skip = 8
    physics_ticks_per_second = 120

    # Set the max in-game time for an episode and compute max steps to do per game
    ep_len_seconds = 20
    max_steps = int(
        round(ep_len_seconds * physics_ticks_per_second / default_tick_skip)
    )

    gym_env = rlgym.make(
        reward_fn=reward_fn,
        obs_builder=UnbiasedObservationBuilder(),
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(max_steps)],
        use_injector=True,
        spawn_opponents=False,
    )

    # Change to SB3 instance wrapper to allow self-play
    # env = SB3SingleInstanceEnv(gym_env)

    # Logger config for training, info outputs to stdout and a csv file

    csv_logger = configure(log_path, ["stdout", "csv"])

    # Wrap env to log reward at each training timestep
    gym_env = Monitor(gym_env, log_path)

    # If a saved model exists, load that and overwrite empty model
    learner = PPO(policy="MlpPolicy", env=gym_env, verbose=1)

    
    try:
        learner = learner.load(saved_model_dir, env=gym_env)
        print("Model Loaded")
    except:
        print("New Model Initialized")

    learner.set_logger(csv_logger)
    # Allows one to stop training and not lose as much progress
    cycles = 40
    for cycle in range(cycles):
        # Learn
        learner.learn(1_000_000)

        print(f"\n\nSaving Model at Cycle #{cycle}\n\n")

        # Save model
        learner.save(saved_model_dir)
