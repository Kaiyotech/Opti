import sys
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from rlgym.envs import Match
from CoyoteObs import CoyoteObsBuilder
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition,\
    NoTouchTimeoutCondition, GoalScoredCondition
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from setter import CoyoteSetter
from CoyoteParser import CoyoteAction
from rewards import ZeroSumReward
from pretrained_agents.necto.necto_v1 import NectoV1
from torch import set_num_threads
from Constants import FRAME_SKIP, ZERO_SUM
from pretrained_agents.nexto.nexto_v2 import NextoV2
import os
set_num_threads(1)


if __name__ == "__main__":
    rew = ZeroSumReward(zero_sum=ZERO_SUM)
    frame_skip = FRAME_SKIP
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
    force_old_deterministic = True
    host = "127.0.0.1"
    if len(sys.argv) > 1:
        host = sys.argv[1]
        if host != "127.0.0.1" and host != "localhost":
            local = False
    if len(sys.argv) > 2:
        name = sys.argv[2]
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
        team_size=3,
        state_setter=CoyoteSetter(),
        obs_builder=CoyoteObsBuilder(expanding=True, tick_skip=FRAME_SKIP, team_size=3),
        action_parser=CoyoteAction(),
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        reward_function=rew,
        tick_skip=frame_skip,
    )

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  db=1,  # testing
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
                  db=1,  # testing
                  )

    model_name = "necto-model-30Y.pt"
    nectov1 = NectoV1(model_string=model_name, n_players=6)
    model_name = "nexto-model.pt"
    nexto = NextoV2(model_string=model_name, n_players=6)

    pretrained_agents = {nectov1: 0, nexto: 0.1}

    RedisRolloutWorker(r, name, match,
                       past_version_prob=past_version_prob,
                       sigma_target=2,
                       evaluation_prob=evaluation_prob,
                       force_paging=True,
                       dynamic_gm=True,
                       send_obs=True,
                       auto_minimize=auto_minimize,
                       send_gamestates=send_gamestate,
                       pretrained_agents=pretrained_agents,
                       gamemode_weights=None,  # {'1v1': 0.3, '2v2': 0.25, '3v3': 0.45}  # testing weights
                       streamer_mode=streamer_mode,
                       deterministic_streamer=deterministic_streamer,
                       force_old_deterministic=force_old_deterministic,
                       ).run()
