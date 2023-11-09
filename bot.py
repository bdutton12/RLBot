import numpy as np
import rlgym
from rlgym.utils.reward_functions.common_rewards import VelocityReward
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.state_setters import DefaultState
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.action_parsers import DefaultAction
from reward_functions import MoveTowardsGoal

from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv, SB3MultipleInstanceEnv

from stable_baselines3.ppo import PPO

"""
Defines a class for RLGym that determines when to end an episode
"""


class TerminalConditions(TerminalCondition):
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        if current_state.orange_score > 3 or current_state.blue_score > 3:
            return True


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
Initialize an instance of Rocket League
"""


def get_match():
    return Match(
        reward_function=VelocityReward(),
        terminal_conditions=[TerminalConditions(), TimeoutCondition(500)],
        obs_builder=UnbiasedObservationBuilder(),
        state_setter=DefaultState(),
        action_parser=DefaultAction(),
        spawn_opponents=True,
    )


if __name__ == "__main__":
    # Make RLGym environment
    gym_env = rlgym.make(
        reward_fn=MoveTowardsGoal(),
        obs_builder=UnbiasedObservationBuilder(),
        terminal_conditions=TerminalConditions(),
        use_injector=True,
        spawn_opponents=True,
    )

    # Change to SB3 instance wrapper to allow self-play
    env = SB3SingleInstanceEnv(gym_env)

    # If a saved model exists, load that and overwrite empty model
    learner = PPO(policy="MlpPolicy", env=env, verbose=1)

    try:
        learner.load("./saved_model/PPO_model.zip", env=env)
        print("Model Loaded")
    except:
        print("New Model Initialized")

    # Learn
    learner.learn(1_000_000)

    # Save model
    learner.save("./saved_model/PPO_model.zip")
