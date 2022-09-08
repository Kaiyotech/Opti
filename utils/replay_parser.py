import numpy as np
import os
from rlgym.utils.common_values import CEILING_Z, BALL_RADIUS, GOAL_HEIGHT
from rlgym.utils.math import euler_to_rotation, cosine_similarity


# curate aerial states with ball and at least one car above 750
def parse_aerial(file_name, _num_cars):
    data = np.load(file_name)
    output = []
    ball_positions = data[:, BALL_POSITION]
    for _i, ball_state in enumerate(ball_positions):
        if ball_state[2] > GOAL_HEIGHT + 100:
            cars = np.split(data[_i][9:], _num_cars)
            for _j in range(_num_cars):
                car_pos = cars[_j][CAR_POS]
                if np.linalg.norm(ball_state - car_pos) < 5 * BALL_RADIUS:
                    output.append(data[_i])
    print(f"Created {len(output)} aerial states from {file_name}")
    output_file = f"aerial_{file_name}"
    if os.path.exists(output_file):
        os.remove(output_file)
    np.save(output_file, output)


# curate flip reset states and save in flip_reset_1v1.npy, etc
def parse_flip_resets(file_name, _num_cars):
    data = np.load(file_name)
    output = []
    ball_positions = data[:, BALL_POSITION]
    for _i, ball_state in enumerate(ball_positions):
        if ball_state[2] > CEILING_Z - ((CEILING_Z - GOAL_HEIGHT) / 2):
            cars = np.split(data[_i][9:], _num_cars)
            for _j in range(_num_cars):
                car_rot = cars[_j][CAR_ROT]
                car_theta = euler_to_rotation(car_rot)
                car_up = car_theta[:, 2]
                car_pos = cars[_j][CAR_POS]
                if np.linalg.norm(ball_state - car_pos) < 3 * BALL_RADIUS \
                                   and cosine_similarity(ball_state - car_pos, -car_up) > 0.7:
                    output.append(data[_i])
    print(f"Created {len(output)} flip reset states from {file_name}")
    output_file = f"flip_resets_{file_name}"
    if os.path.exists(output_file):
        os.remove(output_file)
    np.save(output_file, output)


# curate possible ceiling shot states
def parse_ceiling_shots(file_name, _num_cars):
    data = np.load(file_name)
    output = []
    up = [0, 0, 1]
    for _i, state in enumerate(data):
        cars = np.split(state[9:], _num_cars)
        for _j in range(_num_cars):
            car_rot = cars[_j][CAR_ROT]
            car_theta = euler_to_rotation(car_rot)
            car_up = car_theta[:, 2]
            car_pos = cars[_j][CAR_POS]
            if cosine_similarity(up, -car_up) > 0.9 and car_pos[2] > CEILING_Z - 50:
                output.append(data[_i])

    print(f"Created {len(output)} car ceiling states from {file_name}")
    output_file = f"ceiling_{file_name}"
    if os.path.exists(output_file):
        os.remove(output_file)
    np.save(output_file, output)


# curate kickoff states
def parse_kickoffs(file_name, _num_cars):
    data = np.load(file_name)
    output = []
    ball_positions = data[:, BALL_POSITION]
    for _i, ball_state in enumerate(ball_positions):
        do_it = True
        if ball_state[0] == ball_state[1] == 0:
            cars = np.split(data[_i][9:], _num_cars)
            for _j in range(_num_cars):
                car_pos = cars[_j][CAR_POS]
                if (abs(car_pos[0]) == 2048 and abs(car_pos[1]) == 2560) or \
                        (abs(car_pos[0]) == 256 and abs(car_pos[1]) == 3840) or \
                        (abs(car_pos[0]) == 0 and abs(car_pos[1]) == 4608):
                    do_it = False
                    break
            if do_it:
                output.append(data[_i])

    print(f"Created {len(output)} car kickoff states from {file_name}")
    output_file = f"kickoff_{file_name}"
    if os.path.exists(output_file):
        os.remove(output_file)
    np.save(output_file, output)


BALL_POSITION = slice(0, 3)
BALL_LIN_VEL = slice(3, 6)
BALL_ANG_VEL = slice(6, 9)
CAR_POS = slice(0, 3)
CAR_ROT = slice(3, 6)
CAR_LIN_VEL = slice(6, 9)
CAR_ANG_VEL = slice(9, 12)
CAR_BOOST = slice(12, 13)

input_files = ['ssl_1v1.npy', 'ssl_2v2.npy', 'ssl_3v3.npy']
for i, file in enumerate(input_files):
    num_cars = (i + 1) * 2
    parse_aerial(file, num_cars)
    parse_flip_resets(file, num_cars)
    parse_ceiling_shots(file, num_cars)
    parse_kickoffs(file, num_cars)






