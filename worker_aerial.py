import sys
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from rlgym.envs import Match
from CoyoteObs import CoyoteObsBuilder_Legacy
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition
from mybots_terminals import BallTouchGroundCondition
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from CoyoteParser import CoyoteAction
from rewards import ZeroSumReward
from torch import set_num_threads
from setter import CoyoteSetter
import Constants_aerial
import os

set_num_threads(1)

if __name__ == "__main__":
    frame_skip = Constants_aerial.FRAME_SKIP
    rew = ZeroSumReward(zero_sum=Constants_aerial.ZERO_SUM,
                        goal_w=2,
                        aerial_goal_w=5,
                        double_tap_w=20,
                        flip_reset_w=2,
                        prevent_chain_reset=True,
                        flip_reset_goal_w=4,
                        punish_ceiling_pinch_w=0,
                        punish_backboard_pinch_w=-1,
                        concede_w=-10,
                        velocity_bg_w=0.05,
                        acel_ball_w=0.2,
                        team_spirit=1,
                        cons_air_touches_w=0.01,
                        jump_touch_w=0.1,
                        wall_touch_w=0.5,
                        tick_skip=frame_skip
                        )
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
            auto_minimize = False

    match = Match(
        game_speed=game_speed,
        spawn_opponents=True,
        team_size=team_size,
        state_setter=CoyoteSetter(mode="aerial"),
        obs_builder=CoyoteObsBuilder_Legacy(expanding=True, tick_skip=Constants_aerial.FRAME_SKIP, team_size=team_size,
                                            extra_boost_info=False),
        action_parser=CoyoteAction(),
        terminal_conditions=[GoalScoredCondition(),
                             BallTouchGroundCondition(min_time_sec=0,
                                                      tick_skip=Constants_aerial.FRAME_SKIP,
                                                      time_after_ground_sec=1,
                                                      time_to_arm_sec=2,
                                                      check_towards_goal=True),
                             ],
        reward_function=rew,
        tick_skip=frame_skip,
    )

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  db=Constants_aerial.DB_NUM,
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
                  db=Constants_aerial.DB_NUM,
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
                       gamemode_weights={'1v1': 0.9, '2v2': 0.05, '3v3': 0.05},  # default 1/3
                       streamer_mode=streamer_mode,
                       deterministic_streamer=deterministic_streamer,
                       force_old_deterministic=force_old_deterministic,
                       # testing
                       batch_mode=True,
                       step_size=Constants_aerial.STEP_SIZE,
                       ).run()
