from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z, GOAL_HEIGHT, \
    SIDE_WALL_X, BACK_WALL_Y, CAR_MAX_SPEED, CAR_MAX_ANG_VEL, BALL_RADIUS, BOOST_LOCATIONS
import numpy as np
from numpy import random as rand

DEG_TO_RAD = 3.14159265 / 180


def random_valid_loc() -> np.ndarray:
    rng = np.random.default_rng()
    rand_x = rng.uniform(-4000, 4000)
    if abs(rand_x) > (4096 - 1152):
        rand_y = rng.uniform(-5120 + 1152, 5120 - 1152)
    else:
        rand_y = rng.uniform(-5020, 5020)
    rand_z = rng.uniform(20, 2000)
    return np.asarray([rand_x, rand_y, rand_z])


class BallFrontGoalState(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        rng = np.random.default_rng()

        y_choice = rand.choice([0, 2]) - 1

        state_wrapper.ball.set_pos(x=rng.uniform(-1000, 1000), y=y_choice * rng.uniform(4500, 5000), z=0)
        state_wrapper.ball.set_lin_vel(0, 0, 0)
        state_wrapper.ball.set_ang_vel(0, 0, 0)

        # Loop over every car in the game, skipping 1 since we already did it
        for car in state_wrapper.cars:
            # all cars random
            car.set_pos(rng.uniform(-3500, 3500), rng.uniform(-4400, 4400), 17)
            car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
            car.boost = 0.33


class GroundAirDribble(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        rng = np.random.default_rng()

        car_attack = state_wrapper.cars[0]
        car_defend = None
        for car_y in state_wrapper.cars:
            if car_y.team_num == ORANGE_TEAM:
                car_defend = car_y
        orange_fix = 1
        if rand.choice([0, 1]) and len(state_wrapper.cars) > 1:
            for car_i in state_wrapper.cars:
                if car_i.team_num == ORANGE_TEAM:
                    car_attack = car_i
                    car_defend = state_wrapper.cars[0]  # blue is always 0
                    orange_fix = -1
                    continue

        x_choice = rand.choice([0, 2]) - 1

        rand_x = x_choice * rng.uniform(0, 3000)
        rand_y = rng.uniform(-2000, 2000)
        rand_z = 17
        desired_car_pos = [rand_x, rand_y, rand_z]  # x, y, z
        desired_yaw = (orange_fix * 90 + x_choice * orange_fix * (rng.uniform(5, 15))) * DEG_TO_RAD
        desired_pitch = 0
        desired_roll = 0
        desired_rotation = [desired_pitch, desired_yaw, desired_roll]

        car_attack.set_pos(*desired_car_pos)
        car_attack.set_rot(*desired_rotation)
        car_attack.boost = 100

        car_attack.set_lin_vel(250 * x_choice, 900 * orange_fix, 0)
        car_attack.set_ang_vel(0, 0, 0)

        # put ball in front of car coming towards car at low speed

        ball_x = rand_x
        state_wrapper.ball.set_pos(x=ball_x, y=rand_y + orange_fix * rng.uniform(300, 700),
                                   z=0)
        state_wrapper.ball.set_lin_vel(-80 * x_choice, -800 * orange_fix, 0)
        state_wrapper.ball.set_ang_vel(0, 0, 0)

        # Loop over every car in the game, skipping 1 since we already did it
        for car in state_wrapper.cars:
            if car.id == car_attack.id:
                pass

            # put the defense car in front of ball to block a ground shot, try to force aerial pop
            elif car.id == car_defend.id:
                car.set_pos(ball_x, orange_fix * rng.uniform(3800, 5000), 17)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33
                continue

            # rest of the cars are random
            else:
                car.set_pos(rng.uniform(-1472, 1472), rng.uniform(-1984, 1984), 17)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33


class WallDribble(StateSetter):
    def __init__(
            self,
            max_rand_z=500,
            speed_min=1375,
            speed_max=1425,
    ):
        self.rand_z_max = max_rand_z
        self.speed_min = speed_min
        self.speed_max = speed_max

    def reset(self, state_wrapper: StateWrapper):
        rng = np.random.default_rng()
        # Set up our desired spawn location and orientation for car0 - special one on wall
        # don't at me about the magic numbers, just go with it.
        # blue should aim slightly towards orange goal, and orange towards blue

        car_attack = state_wrapper.cars[0]
        car_defend = None
        for car_y in state_wrapper.cars:
            if car_y.team_num == ORANGE_TEAM:
                car_defend = car_y
        orange_fix = 1
        if rand.choice([0, 1]) and len(state_wrapper.cars) > 1:
            for car_i in state_wrapper.cars:
                if car_i.team_num == ORANGE_TEAM:
                    car_attack = car_i
                    car_defend = state_wrapper.cars[0]  # blue is always 0
                    orange_fix = -1
                    continue

        x_choice = rand.choice([0, 2]) - 1
        rand_x = x_choice * (SIDE_WALL_X - 17)
        rand_y = rng.uniform(-BACK_WALL_Y + 1300, BACK_WALL_Y - 1300)
        rand_z = rng.uniform(100, self.rand_z_max)
        desired_car_pos = [rand_x, rand_y, rand_z]  # x, y, z
        desired_pitch = (90 + orange_fix * (rng.uniform(-20, -5))) * DEG_TO_RAD
        desired_yaw = 90 * DEG_TO_RAD
        desired_roll = 90 * x_choice * DEG_TO_RAD
        desired_rotation = [desired_pitch, desired_yaw, desired_roll]

        car_attack.set_pos(*desired_car_pos)
        car_attack.set_rot(*desired_rotation)
        car_attack.boost = rand.uniform(0.3, 1.0)

        car_attack.set_lin_vel(0, orange_fix * 200 * x_choice, rng.uniform(self.speed_min, self.speed_max))
        car_attack.set_ang_vel(0, 0, 0)

        # Now we will spawn the ball in front of the car_0 with slightly less speed
        # 17 removes the change to move the car to the proper place, so middle of ball is at wall then we move it
        ball_x: np.float32
        if rand_x < 0:
            ball_x = rand_x - 17 + BALL_RADIUS
        else:
            ball_x = rand_x + 17 - BALL_RADIUS
        state_wrapper.ball.set_pos(x=ball_x, y=rand_y + orange_fix * rng.uniform(20, 60),
                                   z=rand_z + rng.uniform(150, 200))
        state_wrapper.ball.set_lin_vel(0, orange_fix * 200, rng.uniform(self.speed_min - 175, self.speed_max - 125))
        state_wrapper.ball.set_ang_vel(0, 0, 0)

        # Loop over every car in the game, skipping 1 since we already did it
        for car in state_wrapper.cars:
            if car.id == car_attack.id:
                pass

            # put the defense car in front of net
            elif car.id == car_defend.id:
                car.set_pos(rng.uniform(-1600, 1600), orange_fix * rng.uniform(3800, 5000), 17)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33
                continue

            # rest of the cars are random
            else:
                car.set_pos(rng.uniform(-1472, 1472), rng.uniform(-1984, 1984), 17)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33


class AirDrag(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        rng = np.random.default_rng()
        # Set up our desired spawn location and orientation for car0 - special one in air
        # don't at me about the magic numbers, just go with it.
        # blue should aim directly towards orange, and vice versa

        car_attack = state_wrapper.cars[0]
        car_defend = None
        for car_y in state_wrapper.cars:
            if car_y.team_num == ORANGE_TEAM:
                car_defend = car_y
        orange_fix = 1
        if rand.choice([0, 1]) and len(state_wrapper.cars) > 1:
            for car_i in state_wrapper.cars:
                if car_i.team_num == ORANGE_TEAM:
                    car_attack = car_i
                    car_defend = state_wrapper.cars[0]  # blue is always 0
                    orange_fix = -1
                    continue

        x_choice = rand.choice([0, 2]) - 1
        rand_x = x_choice * (rng.uniform(0, SIDE_WALL_X - 250))
        rand_y = rng.uniform(-BACK_WALL_Y + 1300, BACK_WALL_Y - 1300)
        rand_z = rng.uniform(300, 800)
        desired_car_pos = [rand_x, rand_y, rand_z]  # x, y, z
        desired_pitch = 20 * DEG_TO_RAD
        desired_yaw = 0  # 90 * DEG_TO_RAD
        desired_roll = 0  # 90 * x_choice * DEG_TO_RAD
        desired_rotation = [desired_pitch, desired_yaw, desired_roll]

        car_attack.set_pos(*desired_car_pos)
        car_attack.set_rot(*desired_rotation)
        car_attack.boost = 100

        car_attack.set_lin_vel(20 * x_choice, rng.uniform(800, 1200), 60)
        car_attack.set_ang_vel(0, 0, 0)

        # Now we will spawn the ball on top of the car matching the velocity
        ball_y: np.float32
        if rand_y < 0:
            ball_y = rand_y - 40
        else:
            ball_y = rand_y + 40
        state_wrapper.ball.set_pos(x=rand_x, y=ball_y + BALL_RADIUS / 2,
                                   z=rand_z + BALL_RADIUS / 2)
        state_wrapper.ball.set_lin_vel(20 * x_choice, rng.uniform(800, 1200), 20)
        state_wrapper.ball.set_ang_vel(0, 0, 0)

        # Loop over every car in the game, skipping 1 since we already did it
        for car in state_wrapper.cars:
            if car.id == car_attack.id:
                pass

            # put the defense car in front of net
            elif car.id == car_defend.id:
                car.set_pos(rng.uniform(-1600, 1600), orange_fix * rng.uniform(3800, 5000), 17)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33
                continue

            # rest of the cars are random
            else:
                car.set_pos(rng.uniform(-1472, 1472), rng.uniform(-1984, 1984), 17)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33


class FlickSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        rng = np.random.default_rng()

        car_attack = state_wrapper.cars[0]
        car_defend = None
        for car_y in state_wrapper.cars:
            if car_y.team_num == ORANGE_TEAM:
                car_defend = car_y
        orange_fix = 1
        if rand.choice([0, 1]) and len(state_wrapper.cars) > 1:
            for car_i in state_wrapper.cars:
                if car_i.team_num == ORANGE_TEAM:
                    car_attack = car_i
                    car_defend = state_wrapper.cars[0]  # blue is always 0
                    orange_fix = -1
                    continue

        x_choice = rand.choice([0, 2]) - 1

        rand_x = int(x_choice * rng.uniform(0, 3000))
        rand_y = int(rng.uniform(-2000, 2000))
        rand_z = 19
        rand_x_vel = rng.uniform(0, 250)
        rand_y_vel = rng.uniform(0, 2000)
        desired_car_pos = [rand_x, rand_y, rand_z]  # x, y, z
        desired_yaw = (orange_fix * 90 + x_choice * orange_fix * (rng.uniform(5, 15))) * DEG_TO_RAD
        desired_pitch = 0
        desired_roll = 0
        desired_rotation = [desired_pitch, desired_yaw, desired_roll]

        car_attack.set_pos(*desired_car_pos)
        car_attack.set_rot(*desired_rotation)
        car_attack.boost = rand.uniform(0, 1)
        desired_car_vel = [rand_x_vel * x_choice, rand_y_vel * orange_fix, 0]
        car_attack.set_lin_vel(*desired_car_vel)
        car_attack.set_ang_vel(0, 0, 0)

        # put ball on top of car, slight random perturbations
        desired_ball_pos = [0, 0, 0]
        desired_ball_pos[0] = desired_car_pos[0] + rng.uniform(-5, 5)
        desired_ball_pos[1] = desired_car_pos[1] + rng.uniform(-5, 5) + orange_fix * 10
        desired_ball_pos[2] = 150 + rng.uniform(-10, 20)
        state_wrapper.ball.set_pos(*desired_ball_pos)
        state_wrapper.ball.set_lin_vel(*desired_car_vel)
        state_wrapper.ball.set_ang_vel(rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2))

        # Loop over every car in the game, skipping 1 since we already did it
        for car in state_wrapper.cars:
            if car.id == car_attack.id:
                pass

            # put the defense car in front of goal
            elif car.id == car_defend.id:
                car.set_pos(rng.uniform(-1600, 1600), orange_fix * rng.uniform(3800, 5000), 17)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33
                continue

            # rest of the cars are random
            else:
                car.set_pos(rng.uniform(-1472, 1472), rng.uniform(-1984, 1984), 17)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33


class RecoverySetter(StateSetter):
    def __init__(self):
        self.rng = np.random.default_rng()
        self.big_boosts = [BOOST_LOCATIONS[i] for i in [3, 4, 15, 18, 29, 30]]
        self.big_boosts = np.asarray(self.big_boosts)
        self.big_boosts[:, -1] = 18
        # self.end_object_tracker = end_object_tracker

    def reset(self, state_wrapper: StateWrapper):

        for car in state_wrapper.cars:
            car.set_pos(*random_valid_loc())
            car.set_rot(self.rng.uniform(-np.pi / 2, np.pi / 2), self.rng.uniform(-np.pi, np.pi),
                        self.rng.uniform(-np.pi, np.pi))
            car.set_lin_vel(self.rng.uniform(-2000, 2000), self.rng.uniform(-2000, 2000), self.rng.uniform(-2000, 2000))
            car.set_ang_vel(self.rng.uniform(-4, 4), self.rng.uniform(-4, 4), self.rng.uniform(-4, 4))
            if self.rng.uniform() > 0.1:
                car.boost = self.rng.uniform(0, 1)
            else:
                car.boost = 0

        # if self.end_object_tracker is not None and self.end_object_tracker[0] != 0:
        loc = random_valid_loc()
        state_wrapper.ball.set_pos(x=loc[0], y=loc[1], z=94)
        state_wrapper.ball.set_lin_vel(0, 0, 0)
        state_wrapper.ball.set_ang_vel(0, 0, 0)


class HalfFlip(StateSetter):
    def __init__(self):
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        assert len(state_wrapper.cars) < 3
        x = self.rng.uniform(-3000, 3000)
        y = self.rng.uniform(-1500, 1500)
        state_wrapper.ball.set_pos(x, y, 94)
        state_wrapper.ball.set_lin_vel(0, 0, 0)
        state_wrapper.ball.set_ang_vel(0, 0, 0)
        mult = -1
        for car in state_wrapper.cars:
            car.set_pos(x, y + mult * 2500)
            car.set_rot(0, mult * np.pi * 0.5, 0)
            car.set_lin_vel(0, 0, 0)
            car.set_ang_vel(0, 0, 0)
            if self.rng.uniform() > 0.1:
                car.boost = self.rng.uniform(0, 1.000001)
            else:
                car.boost = 0
            mult = 1

