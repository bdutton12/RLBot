from typing import *
import gym
import numpy as np

import rlgym
from rlgym.utils.reward_functions.common_rewards import VelocityReward
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState

from model.agent import Agent

"""
Defines a class for RLGym that determines when to end an episode
"""
class CustomTerminalCondition(TerminalCondition):
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        return current_state.last_touch != -1 # End episode when the ball is touched

"""
Observation Object for RLGym

Gets current state of ball and all players and inverts the values if bot is on team Orange
so that the model doesn't need to worry about team
"""
class CustomObsBuilderBluePerspective(ObsBuilder):
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


if __name__ == "__main__":
    # Instantiate RLGym environment
    env = rlgym.make(
        obs_builder=CustomObsBuilderBluePerspective(),
        terminal_conditions=[CustomTerminalCondition()],
        reward_fn=VelocityReward(),
    )

    N = 20
    batch_size = 5
    n_epochs = 4
    eta = 0.0003
    agent = Agent(
        n_actions=env.action_space.n,
        batch_size=batch_size,
        eta=eta,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape,
    )
    n_games = 300

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        # Reset environment to default values for each game (episode)
        observation = env.reset()

        done = False
        score = 0
        while not done:

            # Agent returns a vector [throttle, steer, yaw, pitch, roll, jump, boost, handbrake]
            # The first five are in the range [-1, 1], the latter are boolean
            action, prob, val = agent.choose_action(observation)

            # Updated game state, reward, and whether the episode is done is retrieved
            observation_, reward, done, info = env.step(action)
            
            n_steps += 1
            
            # The reward for all actions in an episode are summed
            score += reward

            # The game state, action taken, probability of success, and reward are stored to memory
            agent.store_to_memory(observation, action, prob, val, reward, done)
            
            # Every N steps, a learning iteration is done
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        
        # Score history is updated and the average of the last 100 scores is taken
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # The model is saved to a file if it is a better solution
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            "episode",
            i,
            "score %.1f" % score,
            "avg score %.1f" % avg_score,
            "time_steps",
            n_steps,
            "learning_steps",
            learn_iters,
        )
