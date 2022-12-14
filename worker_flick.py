import sys
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from rlgym.envs import Match
from CoyoteObs import CoyoteObsBuilder
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from mybots_terminals import BallTouchGroundCondition, PlayerTwoTouch, AttackerTouchCloseGoal
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from CoyoteParser import CoyoteAction
from rewards import ZeroSumReward
from torch import set_num_threads
from setter import CoyoteSetter
import Constants_flick
import os

set_num_threads(1)

if __name__ == "__main__":
    rew = ZeroSumReward(zero_sum=Constants_flick.ZERO_SUM,
                        goal_w=5,
                        concede_w=-5,
                        acel_ball_w=0.5,
                        team_spirit=0,
                        exit_velocity_w=1.5,
                        dribble_w=0,
                        velocity_bg_w=0.01,
                        exit_vel_angle_w=1,
                        min_goal_speed_rewarded_kph=80,
                        )
    frame_skip = Constants_flick.FRAME_SKIP
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    streamer_mode = False
    local = True
    auto_minimize = True
    game_speed = 100
    evaluation_prob = 0
    past_version_prob = 0
    deterministic_streamer = True
    force_old_deterministic = False
    team_size = 3
    dynamic_game = True
    host = "127.0.0.1"
    if len(sys.argv) > 1:
        host = sys.argv[1]
        if host != "127.0.0.1" and host != "localhost":
            local = False
    if len(sys.argv) > 2:
        name = sys.argv[2]
    # if len(sys.argv) > 3 and not dynamic_game:
    #     team_size = int(sys.argv[3])
    if len(sys.argv) > 3:
        if sys.argv[3] == 'GAMESTATE':
            send_gamestate = True
        elif sys.argv[3] == 'STREAMER':
            streamer_mode = True
            evaluation_prob = 0
            game_speed = 1
            deterministic_streamer = True
            auto_minimize = False

    match = Match(
        game_speed=game_speed,
        spawn_opponents=True,
        team_size=team_size,
        state_setter=CoyoteSetter(mode="flick"),
        obs_builder=CoyoteObsBuilder(expanding=True, tick_skip=Constants_flick.FRAME_SKIP, team_size=team_size,
                                     extra_boost_info=True, embed_players=True),
        action_parser=CoyoteAction(),
        terminal_conditions=[GoalScoredCondition(),
                             BallTouchGroundCondition(min_time_sec=0.5,
                                                      tick_skip=Constants_flick.FRAME_SKIP,
                                                      time_after_ground_sec=0.25,
                                                      min_height=100,
                                                      neg_z_check=True),
                             # PlayerTwoTouch(time_to_arm=1, tick_skip=Constants_flick.FRAME_SKIP),
                             TimeoutCondition(fps * 100),
                             AttackerTouchCloseGoal(distance=1000),
                             ],
        reward_function=rew,
        tick_skip=frame_skip,
    )

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  db=Constants_flick.DB_NUM,
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
                  db=Constants_flick.DB_NUM,
                  )

    RedisRolloutWorker(r, name, match,
                       past_version_prob=past_version_prob,
                       sigma_target=2,
                       evaluation_prob=evaluation_prob,
                       force_paging=False,
                       dynamic_gm=dynamic_game,
                       send_obs=True,
                       auto_minimize=auto_minimize,
                       send_gamestates=send_gamestate,
                       gamemode_weights={'1v1': 1, '2v2': 0, '3v3': 0},  # default 1/3
                       streamer_mode=streamer_mode,
                       deterministic_streamer=deterministic_streamer,
                       force_old_deterministic=force_old_deterministic,
                       # testing
                       batch_mode=True,
                       step_size=Constants_flick.STEP_SIZE,
                       ).run()
