import sys
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from rlgym.envs import Match
from CoyoteObs import CoyoteObsBuilder
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, \
    BallTouchedCondition
from mybots_terminals import BallTouchGroundCondition, PlayerTwoTouch, AttackerTouchCloseGoal, ReachObject, \
    PlayerTouchGround
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from CoyoteParser import CoyoteAction
from rewards import ZeroSumReward
from torch import set_num_threads
from setter import CoyoteSetter
from mybots_statesets import Walldash
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym.utils.gamestates.physics_object import PhysicsObject
import Constants_walldash
import os

set_num_threads(1)

if __name__ == "__main__":
    frame_skip = Constants_walldash.FRAME_SKIP
    end_object = PhysicsObject()
    # end_object = None
    end_touched = {"Touched": False}
    rew = ZeroSumReward(zero_sum=Constants_walldash.ZERO_SUM,
                        velocity_pb_w=0.01,
                        vp_end_object_w=0.01,
                        # boost_gain_w=0.35,
                        # boost_spend_w=4,
                        # punish_boost=True,
                        touch_object_w=3.5,
                        touch_ball_w=3.5,
                        boost_remain_touch_w=2,
                        boost_remain_touch_object_w=2,
                        tick_skip=Constants_walldash.FRAME_SKIP,
                        walldash_w=0.15,
                        end_object=end_object,
                        end_touched=end_touched,
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
    deterministic_streamer = False
    force_old_deterministic = False
    # gamemode_weights = {'1v0': 1, '2v2': 0, '3v3': 0}
    team_size = 1
    dynamic_game = False
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
            # gamemode_weights = {'1v1': 1, '2v2': 0, '3v3': 0}
    match = Match(
        game_speed=game_speed,
        spawn_opponents=False,
        team_size=team_size,

        state_setter=WeightedSampleSetter(
            (Walldash(location="back_boost", end_object=end_object, zero_boost_weight=0.7,
                      ball_vel_mult=2.5, min_car_vel=400, max_car_vel=800),
             Walldash(location="45", end_object=end_object, zero_boost_weight=0.7,
                      ball_vel_mult=2.5, min_car_vel=400, max_car_vel=800),
             Walldash(location="90", end_object=end_object, zero_boost_weight=0.7,
                      ball_vel_mult=2.5, min_car_vel=400, max_car_vel=800),
             Walldash(location="same_z", end_object=end_object, zero_boost_weight=0.7,
                      ball_vel_mult=2.5, min_car_vel=400, max_car_vel=800),
             Walldash(location="ball", end_object=end_object, zero_boost_weight=0.7,
                      ball_vel_mult=2.5, min_car_vel=400, max_car_vel=800),
             ),
            (0.35, 0.25, 0.1, 0.15, 0.15)
            # (0.1, 0.4, 0.05, 0.45, 0)  # temp to learn walldash
            # (0.75, 0, 0, 0, 0.25)  # testing
            # (0, 0, 0, 1, 0)  # testing
        ),
        obs_builder=CoyoteObsBuilder(expanding=True,
                                     tick_skip=Constants_walldash.FRAME_SKIP,
                                     team_size=3, extra_boost_info=False,
                                     embed_players=False,
                                     add_jumptime=True,
                                     add_airtime=True,
                                     add_fliptime=True,
                                     add_boosttime=True,
                                     add_handbrake=True,
                                     end_object=end_object),
        action_parser=CoyoteAction(),
        terminal_conditions=[GoalScoredCondition(),
                             TimeoutCondition(fps * 20),
                             # TimeoutCondition(fps * 2),
                             BallTouchedCondition(),
                             ReachObject(end_object=end_object, end_touched=end_touched),
                             PlayerTouchGround(dist_from_side_wall=100, end_object=end_object, allow_boost_full_ground=True),
                             ],
        reward_function=rew,
        tick_skip=frame_skip,
    )

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  db=Constants_walldash.DB_NUM,
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
                  db=Constants_walldash.DB_NUM,
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
                       # gamemode_weights=gamemode_weights,  # default 1/3
                       streamer_mode=streamer_mode,
                       deterministic_streamer=deterministic_streamer,
                       force_old_deterministic=force_old_deterministic,
                       # testing
                       batch_mode=True,
                       step_size=Constants_walldash.STEP_SIZE,
                       ).run()
