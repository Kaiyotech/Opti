import sys
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from rlgym.envs import Match
from CoyoteObs import CoyoteObsBuilder
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, \
    NoTouchTimeoutCondition
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from CoyoteParser import CoyoteAction
from rewards import ZeroSumReward
from torch import set_num_threads
from setter import CoyoteSetter
import Constants_gp
import os

set_num_threads(1)

if __name__ == "__main__":
    rew = ZeroSumReward(zero_sum=Constants_gp.ZERO_SUM,
                        goal_w=10,
                        concede_w=-10,
                        # double_tap_w=5,
                        velocity_bg_w=0.075,
                        velocity_pb_w=0,
                        boost_gain_w=0.3,
                        punish_boost=True,
                        # boost_spend_w=2.25,
                        demo_w=0.5,
                        acel_ball_w=1,
                        team_spirit=1,
                        # cons_air_touches_w=2,
                        jump_touch_w=1,
                        wall_touch_w=0.25,
                        touch_grass_w=0, )
    frame_skip = Constants_gp.FRAME_SKIP
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    streamer_mode = False
    local = True
    auto_minimize = True
    game_speed = 100
    evaluation_prob = 0.01
    past_version_prob = 0.1
    deterministic_streamer = True
    force_old_deterministic = False
    batch_mode = True
    team_size = 3
    dynamic_game = True
    infinite_boost_odds = 0.1
    host = "127.0.0.1"
    if len(sys.argv) > 1:
        host = sys.argv[1]
        if host != "127.0.0.1" and host != "localhost":
            local = False
            batch_mode = False
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
            infinite_boost_odds = 0.1

    match = Match(
        game_speed=game_speed,
        spawn_opponents=True,
        team_size=team_size,
        state_setter=CoyoteSetter(mode="normal"),
        obs_builder=CoyoteObsBuilder(expanding=True, tick_skip=Constants_gp.FRAME_SKIP, team_size=team_size,
                                     extra_boost_info=True, embed_players=True,
                                     infinite_boost_odds=infinite_boost_odds),
        action_parser=CoyoteAction(),
        terminal_conditions=[GoalScoredCondition(),
                             NoTouchTimeoutCondition(fps * 15),
                             TimeoutCondition(fps * 300),
                             ],
        reward_function=rew,
        tick_skip=frame_skip,
    )

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  db=Constants_gp.DB_NUM,
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
                  db=Constants_gp.DB_NUM,
                  )

    worker = RedisRolloutWorker(r, name, match,
                                past_version_prob=past_version_prob,
                                sigma_target=2,
                                evaluation_prob=evaluation_prob,
                                force_paging=False,
                                dynamic_gm=dynamic_game,
                                send_obs=True,
                                auto_minimize=auto_minimize,
                                send_gamestates=send_gamestate,
                                gamemode_weights={'1v1': 0.20, '2v2': 0.35, '3v3': 0.45},  # default 1/3
                                streamer_mode=streamer_mode,
                                deterministic_streamer=deterministic_streamer,
                                force_old_deterministic=force_old_deterministic,
                                # testing
                                batch_mode=batch_mode,
                                step_size=Constants_gp.STEP_SIZE,
                                )

    worker.env._match._obs_builder.env = worker.env

    worker.run()
