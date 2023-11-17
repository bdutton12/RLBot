import numpy as np
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards import (
    AlignBallGoal,
    VelocityBallToGoalReward,
)
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import common_values, math
from rlgym.utils.reward_functions.common_rewards import (
    LiuDistancePlayerToBallReward,
    VelocityBallToGoalReward,
    VelocityReward,
    FaceBallReward,
    AlignBallGoal,
    EventReward,
)
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    TouchBallReward,
    VelocityPlayerToBallReward,
)

"""
Reward function that rewards facing the ball, aligning the ball with the net, and touching the ball
towards the opposing net
"""


class AlignAndDistanceReward(RewardFunction):
    def __init__(
        self,
        dist_weight=0.7,
        align_ball_weight=0.3,
        face_ball_weight=0.4,
        touch_ball_weight=0.9,
        ball_to_goal_weight=0.6,
        player_vel_weight=0.4,
        goal_reward=1000,
        shot_reward=100,
        save_reward=300,
        demo_reward=50,
    ):
        self.dist_weight = dist_weight
        self.align_ball_weight = align_ball_weight
        self.face_ball_weight = face_ball_weight
        self.touch_ball_weight = touch_ball_weight
        self.ball_to_goal_weight = ball_to_goal_weight
        self.player_velocity_weight = player_vel_weight

        self.event_reward_func = EventReward(
            goal_reward, 0, -goal_reward, 50, shot_reward, save_reward, demo_reward, 10
        )
        self.dist_func = LiuDistancePlayerToBallReward()
        self.align_ball_func = AlignBallGoal()
        self.face_ball_func = FaceBallReward()
        self.touch_ball_func = TouchBallReward()
        self.ball_to_goal_func = VelocityBallToGoalReward()
        self.player_velocity_func = VelocityReward()

    def reset(self, initial_state: GameState):
        self.dist_func.reset(initial_state)
        self.align_ball_func.reset(initial_state)
        self.face_ball_func.reset(initial_state)
        self.touch_ball_func.reset(initial_state)
        self.ball_to_goal_func.reset(initial_state)
        self.player_velocity_func.reset(initial_state)
        self.event_reward_func.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # Reward for being close to the ball
        reward = self.dist_weight * self.dist_func.get_reward(
            player, state, previous_action
        )

        # Reward for centering the ball in front of opponent's net
        reward += self.align_ball_weight * self.align_ball_func.get_reward(
            player, state, previous_action
        )

        # Reward for facing the opponent net with ball between
        reward += self.face_ball_weight * self.face_ball_func.get_reward(
            player, state, previous_action
        )

        reward += self.touch_ball_weight * self.touch_ball_func.get_reward(
            player, state, previous_action
        )

        # Reward for velocity of ball towards opponent goal
        reward += self.ball_to_goal_weight * self.ball_to_goal_func.get_reward(
            player, state, previous_action
        )

        reward += self.player_velocity_weight * self.player_velocity_func.get_reward(
            player, state, previous_action
        )

        # Reward for goals, saves, shots, and demos
        reward += self.event_reward_func.get_reward(player, state, previous_action)

        return reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.dist_weight * self.dist_func.get_final_reward(
            player, state, previous_action
        )
        reward += self.align_ball_weight * self.align_ball_func.get_final_reward(
            player, state, previous_action
        )
        reward += self.face_ball_weight * self.face_ball_func.get_final_reward(
            player, state, previous_action
        )
        reward += self.touch_ball_weight * self.align_ball_func.get_final_reward(
            player, state, previous_action
        )
        reward += self.ball_to_goal_weight * self.ball_to_goal_func.get_final_reward(
            player, state, previous_action
        )
        reward += (
            self.player_velocity_weight
            * self.player_velocity_func.get_final_reward(player, state, previous_action)
        )
        reward += self.event_reward_func.get_final_reward(
            player, state, previous_action
        )

        return reward


class HybridReward(RewardFunction):
    reward_functions: list[RewardFunction]

    def __init__(self):
        super().__init__()

        self.reward_functions = [
            VelocityReward(),
            VelocityBallToGoalReward(),
        ]
        self.reward_weights = np.array([1, 1])

        assert len(self.reward_functions) == len(self.reward_weights)

    def reset(self, initial_state: GameState) -> None:
        for func in self.reward_functions:
            func.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        return float(np.dot(self.reward_weights, rewards))

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        return float(np.dot(self.reward_weights, rewards))
