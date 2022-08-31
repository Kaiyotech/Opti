from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z, GOAL_HEIGHT,\
    SIDE_WALL_X, BACK_WALL_Y, CAR_MAX_SPEED, CAR_MAX_ANG_VEL, BALL_RADIUS
import numpy as np
from numpy import random as rand

DEG_TO_RAD = 3.14159265 / 180


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
                car.set_pos(ball_x, orange_fix * rng.uniform(3800, 5000), 0)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33
                continue

            # rest of the cars are random
            else:
                car.set_pos(rng.uniform(-1472, 1472), rng.uniform(-1984, 1984), 0)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33


class WallDribble(StateSetter):
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
        rand_z = rng.uniform(100, 500)
        desired_car_pos = [rand_x, rand_y, rand_z]  # x, y, z
        desired_pitch = (90 + orange_fix * (rng.uniform(-20, -5))) * DEG_TO_RAD
        desired_yaw = 90 * DEG_TO_RAD
        desired_roll = 90 * x_choice * DEG_TO_RAD
        desired_rotation = [desired_pitch, desired_yaw, desired_roll]

        car_attack.set_pos(*desired_car_pos)
        car_attack.set_rot(*desired_rotation)
        car_attack.boost = 100

        car_attack.set_lin_vel(0, orange_fix * 200 * x_choice, rng.uniform(1375, 1425))
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
        state_wrapper.ball.set_lin_vel(0, orange_fix * 200, rng.uniform(1200, 1300))
        state_wrapper.ball.set_ang_vel(0, 0, 0)

        # Loop over every car in the game, skipping 1 since we already did it
        for car in state_wrapper.cars:
            if car.id == car_attack.id:
                pass

            # put the defense car in front of net
            elif car.id == car_defend.id:
                car.set_pos(rng.uniform(-1600, 1600), orange_fix * rng.uniform(3800, 5000), 0)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33
                continue

            # rest of the cars are random
            else:
                car.set_pos(rng.uniform(-1472, 1472), rng.uniform(-1984, 1984), 0)
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
                car.set_pos(rng.uniform(-1600, 1600), orange_fix * rng.uniform(3800, 5000), 0)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33
                continue

            # rest of the cars are random
            else:
                car.set_pos(rng.uniform(-1472, 1472), rng.uniform(-1984, 1984), 0)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33
