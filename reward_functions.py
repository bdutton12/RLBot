from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.reward_functions.misc_rewards import AlignBallGoal
from rlgym.utils.reward_functions.player_ball_rewards import FaceBallReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import TouchBallReward
import numpy as np

'''
Reward function that rewards facing the ball, aligning the ball with the net, and touching the ball
towards the opposing net
'''
class AlignAndDistanceReward(RewardFunction):
  def __init__(self, dist_weight=0.7, align_ball_weight=0.5, face_ball_weight=0.4, touch_ball_weight=0.9):
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

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    dist_reward = self.dist_weight * self.dist_func.get_reward(player, state, previous_action)
    align_ball_reward = self.align_ball_weight * self.align_ball_func.get_reward(player, state, previous_action)
    face_ball_reward = self.face_ball_weight * self.face_ball_func.get_reward(player, state, previous_action)
    touch_ball_reward = self.touch_ball_weight * self.align_ball_func.get_reward(player, state, previous_action)
    
    return dist_reward + align_ball_reward + face_ball_reward + touch_ball_reward
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    dist_reward = self.dist_weight * self.dist_func.get_final_reward(player, state, previous_action)
    align_ball_reward = self.align_ball_weight * self.align_ball_func.get_final_reward(player, state, previous_action)
    face_ball_reward = self.face_ball_weight * self.face_ball_func.get_final_reward(player, state, previous_action)
    touch_ball_reward = self.touch_ball_weight * self.align_ball_func.get_final_reward(player, state, previous_action)
    
    return dist_reward + align_ball_reward + face_ball_reward + touch_ball_reward