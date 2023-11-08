import gym
import numpy as np

import rlgym
from rlgym.utils.reward_functions.common_rewards import VelocityReward
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.state_setters import DefaultState
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState

from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from stable_baselines3 import PPO


"""
Defines a class for RLGym that determines when to end an episode
"""


class TerminalConditions(TerminalCondition):
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        return current_state.last_touch != -1  # End episode when the ball is touched


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
    ) -> Any:
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
Initialize an instance of Rocket League
"""
def get_match():
    return Match(
        reward_function=VelocityReward(),
        terminal_conditions=[TerminalConditions()],
        obs_builder=UnbiasedObservationBuilder(),
        state_setter=DefaultState(),
        self_play=True,
    )


if __name__ == "__main__":
    # Make the default rlgym environment

    # Initialize PPO from stable_baselines3
    env = SB3MultipleInstanceEnv(
        match_func_or_matches=get_match, num_instances=2, wait_time=20
    )

    learner = PPO(policy="MlpPolicy", env=env, verbose=1)
    learner.load("./saved_model/PPO_model.zip", env=env)
    learner.learn(1_000_000)
    learner.save("./saved_model/PPO_model.zip")
