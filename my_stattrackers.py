import numpy as np

from rocket_learn.utils.gamestate_encoding import StateConstants
from rocket_learn.utils.stat_trackers.stat_tracker import StatTracker


class ExitVelocity(StatTracker):
    def __init__(self):
        super().__init__("average_ball_speed")
        self.count = 0
        self.total_speed = 0.0

    def reset(self):
        return NotImplemented
        self.count = 0
        self.total_speed = 0.0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        ball_speeds = gamestates[:, StateConstants.BALL_LINEAR_VELOCITY]
        xs = ball_speeds[:, 0]
        ys = ball_speeds[:, 1]
        zs = ball_speeds[:, 2]
        speeds = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
        self.total_speed += np.sum(speeds)
        self.count += speeds.size

    def get_stat(self):
        return self.total_speed / (self.count or 1)


class GoalSpeedTop5perc(StatTracker):
    def __init__(self):
        super().__init__("avg_goal_speed_top_5_percent")
        self.count = 0
        self.goal_speeds = None

    def reset(self):
        self.count = 0
        self.goal_speeds = []

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        if gamestates.ndim > 1 and len(gamestates) > 1:
            end = gamestates[-2]
            goal_speed = end[StateConstants.BALL_LINEAR_VELOCITY]
            goal_speed = np.linalg.norm(goal_speed)

            self.goal_speeds.append(goal_speed / 27.78)  # convert to km/h
            self.count += 1

    def get_stat(self):
        self.goal_speeds.sort()
        top_5 = int(-1 * self.count / 20)
        self.goal_speeds = self.goal_speeds[top_5:]
        total_speed = sum(self.goal_speeds)
        return total_speed / (top_5 or 1)
