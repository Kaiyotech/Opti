from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState
from rlgym.utils.common_values import BALL_RADIUS, BACK_WALL_Y, CEILING_Z


def ball_towards_goal(ball):
    x, y, z = ball.position
    vx, vy, vz = ball.linear_velocity
    return y * vy > 0 and (abs(x) < 1000 or x * vx < 0) and abs(y) > 4000


class BallTouchGroundCondition(TerminalCondition):
    """
    A condition that will terminate an episode after ball touches ground
    """

    def __init__(self, min_time_sec=3, tick_skip=8, time_after_ground_sec=0, min_height=2 * BALL_RADIUS,
                 neg_z_check=False,
                 check_towards_goal=False,
                 time_to_arm_sec=0,
                 ):
        super().__init__()
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
            if not self.neg_z_check and not self.check_towards_goal:
                return True
            elif self.neg_z_check:
                return current_state.ball.linear_velocity[2] < -1
            elif self.check_towards_goal:
                return not ball_towards_goal(current_state.ball)
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

