import numpy as np
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import common_values, math
from rlgym.utils.reward_functions.common_rewards import (
    LiuDistancePlayerToBallReward,
    VelocityBallToGoalReward,
    VelocityReward,
    FaceBallReward,
    AlignBallGoal,
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
        align_ball_weight=0.5,
        face_ball_weight=0.4,
        touch_ball_weight=0.9,
    ):
        self.dist_weight = dist_weight
        self.align_ball_weight = align_ball_weight
        self.face_ball_weight = face_ball_weight
        self.touch_ball_weight = touch_ball_weight

        self.dist_func = LiuDistancePlayerToBallReward()
        self.align_ball_func = AlignBallGoal()
        self.face_ball_func = FaceBallReward()
        self.touch_ball_func = TouchBallReward()

    def reset(self, initial_state: GameState):
        self.dist_func.reset(initial_state)
        self.align_ball_func.reset(initial_state)
        self.face_ball_func.reset(initial_state)
        self.touch_ball_func.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        dist_reward = self.dist_weight * self.dist_func.get_reward(
            player, state, previous_action
        )
        align_ball_reward = self.align_ball_weight * self.align_ball_func.get_reward(
            player, state, previous_action
        )
        face_ball_reward = self.face_ball_weight * self.face_ball_func.get_reward(
            player, state, previous_action
        )
        touch_ball_reward = self.touch_ball_weight * self.align_ball_func.get_reward(
            player, state, previous_action
        )

        return dist_reward + align_ball_reward + face_ball_reward + touch_ball_reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        dist_reward = self.dist_weight * self.dist_func.get_final_reward(
            player, state, previous_action
        )
        align_ball_reward = (
            self.align_ball_weight
            * self.align_ball_func.get_final_reward(player, state, previous_action)
        )
        face_ball_reward = self.face_ball_weight * self.face_ball_func.get_final_reward(
            player, state, previous_action
        )
        touch_ball_reward = (
            self.touch_ball_weight
            * self.align_ball_func.get_final_reward(player, state, previous_action)
        )

        return dist_reward + align_ball_reward + face_ball_reward + touch_ball_reward


class MoveTowardsGoal(RewardFunction):
    def __init__(
        self,
        player_velocity_weight=1,
        ball_velocity_weight=1,
        goal_reward=100,
        shot_reward=5,
        save_reward=30,
        demo_reward=10,
    ):
        self.player_v = player_velocity_weight
        self.ball_v = ball_velocity_weight
        self.goal = goal_reward
        self.shot = shot_reward
        self.save = save_reward
        self.demo = demo_reward

        self.prev_goals = 0
        self.prev_shots = 0
        self.prev_saves = 0
        self.prev_demos = 0

        self.blue_goals = 0
        self.orange_goals = 0

        self.vel_to_goal = VelocityBallToGoalReward()
        self.vel = VelocityReward()

    def reset(self, initial_state: GameState):
        self.vel_to_goal.reset()
        self.vel.reset()

        self.prev_goals = 0
        self.prev_shots = 0
        self.prev_saves = 0
        self.prev_demos = 0

        self.blue_goals = 0
        self.orange_goals = 0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # Reward for player velocity and ball velocity
        ball_velocity_reward = self.ball_v * self.vel_to_goal.get_reward(
            player, state, previous_action
        )

        player_velocity_reward = self.player_v * self.vel.get_reward(
            player, state, previous_action
        )

        reward = ball_velocity_reward + player_velocity_reward

        # Reward for goals, saves, shots, and demos
        if self.prev_goals < player.match_goals:
            reward += self.goal
            self.prev_goals = player.match_goals

        if self.prev_saves < player.match_saves:
            reward += self.save
            self.prev_saves = player.match_saves

        if self.prev_shots < player.match_shots:
            reward += self.shot
            self.prev_shots = player.match_shots

        if self.prev_demos < player.match_demolishes:
            reward += self.demo
            self.prev_demos = player.match_demolishes

        # Penalize for getting scored on
        if (
            player.team_num == common_values.ORANGE_TEAM
            and state.blue_score > self.blue_goals
        ) or (
            player.team_num == common_values.BLUE_TEAM
            and state.orange_score > self.orange_goals
        ):
            reward -= self.goal
            self.blue_goals = state.blue_score
            self.orange_goals = state.orange_score

        return reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return 0
