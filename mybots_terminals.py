from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState
from rlgym.utils.common_values import BALL_RADIUS, BACK_WALL_Y, CEILING_Z, BOOST_LOCATIONS, SIDE_WALL_X
from rlgym.utils.gamestates.physics_object import PhysicsObject
import numpy as np


def ball_towards_goal(ball, y_distance_check, allow_pinch_cont):
    x, y, z = ball.position
    vx, vy, vz = ball.linear_velocity
    y_distance = BACK_WALL_Y - y_distance_check
    normal_check = y * \
        vy > 0 and (abs(x) < 1000 or x * vx < 0) and abs(y) > y_distance
    if not allow_pinch_cont:
        return normal_check
    else:
        pinch_check = (abs(x) > 3500 and x * vx > 0) or (abs(x)
                                                         < 3800 and abs(y) > 4600 and y * vy > 0)
        return normal_check or pinch_check


class BallTouchGroundCondition(TerminalCondition):
    """
    A condition that will terminate an episode after ball touches ground
    """

    def __init__(self, min_time_sec=3, tick_skip=8, time_after_ground_sec=0, min_height=2 * BALL_RADIUS,
                 neg_z_check=False,
                 check_towards_goal=False,
                 time_to_arm_sec=0,
                 y_distance_goal=1120,
                 on_ground_again=False,
                 allow_pinch_cont=False,
                 ):
        super().__init__()
        self.allow_pinch_cont = allow_pinch_cont
        self.on_ground_again = on_ground_again
        self.y_distance_goal = y_distance_goal
        self.min_steps = min_time_sec * (120 // tick_skip)
        self.time_to_arm_steps = time_to_arm_sec * (120 // tick_skip)
        self.steps = 0
        self.steps_after_ground = time_after_ground_sec * (120 // tick_skip)
        self.touch_time = 0
        self.touched = False
        self.min_height = min_height
        self.neg_z_check = neg_z_check
        self.check_towards_goal = check_towards_goal
        assert not (neg_z_check and check_towards_goal)

    def reset(self, initial_state: GameState):
        self.steps = 0
        self.touch_time = 0
        self.touched = False

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if ball is touching the ground and it has been minimum number of steps and it has been a number
        of seconds since the ball first touched the ground.
        """
        self.steps += 1
        if not self.touched and current_state.ball.position[2] < self.min_height and self.steps > self.time_to_arm_steps:
            self.touch_time = self.steps
            self.touched = True
        if self.touched and self.steps > self.min_steps and self.steps > self.touch_time + self.steps_after_ground:
            if self.on_ground_again and not current_state.ball.position[2] < self.min_height:
                return False
            if not self.neg_z_check and not self.check_towards_goal:
                return True
            elif self.neg_z_check:
                return current_state.ball.linear_velocity[2] < -1
            elif self.check_towards_goal:
                return not ball_towards_goal(current_state.ball, self.y_distance_goal, self.allow_pinch_cont)
        else:
            return False


class KickoffTrainer(TerminalCondition):
    """
    A condition that triggers half a second after first touch
    """

    def __init__(self, min_time_sec=0.5, tick_skip=8):
        super().__init__()
        self.min_steps = min_time_sec * (120 // tick_skip)
        self.steps = 0
        self.touch_steps = 0

    def reset(self, initial_state: GameState):
        self.steps = 0
        self.touch_steps = 10_000_000
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if ball is touching the ground and it has been minimum number of steps
        """
        self.steps += 1
        for player in current_state.players:
            if player.ball_touched:
                self.touch_steps = self.steps
                break
        if self.steps > self.touch_steps + self.min_steps:
            return True
        elif (current_state.ball.position[1] > (BACK_WALL_Y - 2 * BALL_RADIUS) and
              current_state.ball.linear_velocity[1] < 0) or (current_state.ball.position[1] <
                                                             (-1 * BACK_WALL_Y + 2 * BALL_RADIUS) and current_state.ball.linear_velocity[1] > 0):
            return True
        else:
            return False


class BallStopped(TerminalCondition):
    """
    A condition that triggers after ball touches ground after min_time_sec after first touch
    """

    def __init__(self, min_time_sec=5, tick_skip=8, max_time_sec=10_000_000):
        super().__init__()
        self.min_steps = min_time_sec * (120 // tick_skip)
        self.steps = 0
        self.max_steps = max_time_sec * (120 // tick_skip)
        # self.touch_steps = 0

    def reset(self, initial_state: GameState):
        self.steps = 0
        # self.touch_steps = 10_000_000
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if ball is touching the ground and stopped and it has been minimum number of steps
        """
        self.steps += 1
        # for player in current_state.players:
        #     if player.ball_touched:
        #         self.touch_steps = self.steps
        #         break
        if self.steps > self.max_steps:
            return True
        if self.steps > self.min_steps and current_state.ball.position[2] < (2 * BALL_RADIUS) \
            and current_state.ball.linear_velocity[0] == 0 and current_state.ball.linear_velocity[1] == 0 and \
                current_state.ball.linear_velocity[2] == 0:
            return True
        else:
            return False


class PlayerTwoTouch(TerminalCondition):
    """
    A condition that triggers after ball touches ground after min_time_sec after first touch
    """

    def __init__(self, time_to_arm=0.25, tick_skip=8):
        super().__init__()
        self.time_to_arm_steps = time_to_arm * (120 // tick_skip)
        self.steps = 0
        self.toucher = -1
        self.touched = False
        self.no_touch_steps = 0
        self.armed = False

    def reset(self, initial_state: GameState):
        self.steps = 0
        self.toucher = -1
        self.touched = False
        self.no_touch_steps = 10_000_000
        self.armed = False

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if first player to touch the ball stops touching it for time_to_arm seconds and then touches again
        """
        self.steps += 1
        if not self.touched:
            for i, player in enumerate(current_state.players):
                if player.ball_touched:
                    self.toucher = i
                    self.touched = True
                    return False
        else:
            if not current_state.players[self.toucher].ball_touched and not self.armed:
                self.no_touch_steps = self.steps
                self.armed = True
            elif (self.steps > self.no_touch_steps + self.time_to_arm_steps) and \
                    current_state.players[self.toucher].ball_touched:
                return True
            elif current_state.players[self.toucher].ball_touched:
                self.armed = False
                self.no_touch_steps = 10_000_000
        return False


class BallTouchCeilingCondition(TerminalCondition):
    """
    A condition that will terminate an episode after ball touches ceiling
    """

    def __init__(self, ):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if ball is close to ceiling
        """

        return current_state.ball.position[2] > (CEILING_Z - 120)


class AttackerTouchCloseGoal(TerminalCondition):
    """
    A condition that triggers after ball touches ground after min_time_sec after first touch
    """

    def __init__(self, distance=1000):
        super().__init__()
        self.toucher = -1
        self.touched = None
        self.y_limit = None
        self.distance = distance
        self.touch_team = None

    def reset(self, initial_state: GameState):
        self.toucher = -1
        self.touched = False
        self.y_limit = None
        self.touch_team = None

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if first player to touch the ball stops touching it for time_to_arm seconds and then touches again
        """
        if not self.touched:
            for i, player in enumerate(current_state.players):
                if player.ball_touched:
                    self.toucher = i
                    self.touched = True
                    mid = len(current_state.players) // 2
                    if i < mid:
                        self.y_limit = 5120 - self.distance
                        self.touch_team = 0
                    else:
                        self.y_limit = -5120 + self.distance
                        self.touch_team = 1
                    return False
        elif current_state.players[self.toucher].ball_touched:
            if current_state.ball.position[1] > self.y_limit and self.touch_team == 0:
                return True
            elif current_state.ball.position[1] < self.y_limit and self.touch_team == 1:
                return True
        return False


class ReachObject(TerminalCondition):
    """
    A condition that triggers after ball touches ground after min_time_sec after first touch
    """

    def __init__(self, end_object: PhysicsObject, end_touched: dict):
        super().__init__()
        self.end_touched = end_touched
        self.end_object = end_object

    def reset(self, initial_state: GameState):
        self.end_touched["Touched"] = False

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if a player reaches the objective
        """
        for i, player in enumerate(current_state.players):
            # Octane hitbox offset is np.array([13.88, 0, 20.75])
            # Octane hitbox dimensions are np.array([118.01, 84.2, 36.16])
            ball_local_pos = np.matmul(np.transpose(player.car_data.rotation_mtx(
            )), self.end_object.position - player.car_data.position) - np.array([13.88, 0, 20.75])
            cp = np.copy(ball_local_pos)
            car_half_extent = np.array([118.01, 84.2, 36.16]) / 2
            for i in range(3):
                cp[i] = min(car_half_extent[i], cp[i])
                cp[i] = max(-car_half_extent[i], cp[i])
            # ball collision radius is 93.15, car hitbox margin is 0.04, summing to 93.19
            intersectionDist = 93.19
            nv = ball_local_pos - cp
            if np.dot(nv, nv) < intersectionDist ** 2:  # touched the objective
                self.end_touched["Touched"] = True
                return True


class PlayerTouchGround(TerminalCondition):

    def __init__(self, dist_from_side_wall: int = -50, end_object: PhysicsObject = None, allow_boost_full_ground=False):
        super().__init__()
        self.allow_boost_full_ground = allow_boost_full_ground
        self.end_object = end_object
        self.dist_from_side_wall = dist_from_side_wall

    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if a player touches ground, with hacks for end object allowances
        """
        dist_limit_x = self.dist_from_side_wall
        if self.end_object is not None and (
                abs(current_state.ball.position[0]) == 3072 and abs(current_state.ball.position[1]) == 4096):
            if self.allow_boost_full_ground:
                return False
            dist_limit_x = 1300  # allow reaching boost
        elif self.end_object is not None and \
                self.end_object.position[0] == self.end_object.position[1] == self.end_object.position[2] == -1:
            return False

        for i, player in enumerate(current_state.players):
            if player.on_ground and player.car_data.position[2] < 22:
                return (SIDE_WALL_X - abs(player.car_data.position[0])) > dist_limit_x
