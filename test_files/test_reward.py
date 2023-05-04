from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils import math
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np


# THIS REWARD IS FOR DEMONSTRATION PURPOSES ONLY
class TestReward(RewardFunction):
    def __init__(self):
        super().__init__()


    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, previous_model_action) -> float:
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, previous_model_action) -> float:
        return 0
