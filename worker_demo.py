import sys
from redis import Redis
from redis.retry import Retry  # noqa
from redis.backoff import ExponentialBackoff  # noqa
from redis.exceptions import ConnectionError, TimeoutError

from CoyoteObs import CoyoteObsBuilder
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from my_matchmaker import MatchmakerFullVPretrained
from CoyoteParser import CoyoteAction
from rewards import ZeroSumReward
from torch import set_num_threads
from setter import CoyoteSetter
import Constants_demo
import os

set_num_threads(1)

if __name__ == "__main__":
    rew = ZeroSumReward(zero_sum=Constants_demo.ZERO_SUM,
                        demo_w=6,
                        got_demoed_w=-6,
                        goal_w=5,
                        concede_w=-5,
                        velocity_pb_w=0.01,
                        velocity_bg_w=0.02,
                        velocity_po_w=0.001,
                        vel_po_mult_ss=5,
                        vel_po_mult_neg=0.01,
                        boost_gain_w=0.5,
                        boost_spend_w=-0.3,
                        tick_skip=Constants_demo.FRAME_SKIP,
                        )
    frame_skip = Constants_demo.FRAME_SKIP
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    streamer_mode = False
    local = True
    auto_minimize = True
    game_speed = 100
    evaluation_prob = 0
    past_version_prob = 0  # 0.5  # 0.1
    non_latest_version_prob = [1, 0, 0,
                               0]  # [0.825, 0.0826, 0.0578, 0.0346]  # this includes past_version and pretrained
    deterministic_streamer = True
    force_old_deterministic = False
    gamemode_weights = {'1v1': 0.8, '2v2': 0.1, '3v3': 0.1}
    visualize = False
    simulator = True
    batch_mode = True
    team_size = 3
    dynamic_game = True
    infinite_boost_odds = 0.2
    host = "127.0.0.1"
    epic_rl_exe_path = None  # "D:/Program Files/Epic Games/rocketleague_old/Binaries/Win64/RocketLeague.exe"

    pretrained_agents = Constants_demo.pretrained_agents

    matchmaker = MatchmakerFullVPretrained(pretrained_agents=pretrained_agents)

    if len(sys.argv) > 1:
        host = sys.argv[1]
        if host != "127.0.0.1" and host != "localhost":
            local = False
            batch_mode = False
            epic_rl_exe_path = None
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
            # infinite_boost_odds = 0.2
            simulator = False
            past_version_prob = 0

            # gamemode_weights = {'1v0': 0.4, '1v1': 0.6}

        elif sys.argv[3] == 'VISUALIZE':
            visualize = True

    if simulator:
        from rlgym_sim.envs import Match as Sim_Match
        from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, \
            NoTouchTimeoutCondition
    else:
        from rlgym.envs import Match
        from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, \
            NoTouchTimeoutCondition

    setter = CoyoteSetter(mode="demo", simulator=simulator)

    match = Match(
        game_speed=game_speed,
        spawn_opponents=True,
        team_size=team_size,
        state_setter=setter,
        obs_builder=CoyoteObsBuilder(expanding=True, tick_skip=Constants_demo.FRAME_SKIP, team_size=team_size,
                                     extra_boost_info=False, embed_players=False,
                                     ),
        action_parser=CoyoteAction(),
        terminal_conditions=[GoalScoredCondition(),
                             TimeoutCondition(fps * 20),
                             ],
        reward_function=rew,
        tick_skip=frame_skip,
    ) if not simulator else Sim_Match(
        spawn_opponents=True,
        team_size=team_size,
        state_setter=setter,
        obs_builder=CoyoteObsBuilder(expanding=True, tick_skip=Constants_demo.FRAME_SKIP, team_size=team_size,
                                     extra_boost_info=False, embed_players=False,
                                     ),
        action_parser=CoyoteAction(),
        terminal_conditions=[GoalScoredCondition(),
                             TimeoutCondition(fps * 20),
                             ],
        reward_function=rew,
    )

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  db=Constants_demo.DB_NUM,
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
                  db=Constants_demo.DB_NUM,
                  )

    worker = RedisRolloutWorker(r, name, match,
                                matchmaker=matchmaker,
                                sigma_target=1,
                                evaluation_prob=evaluation_prob,
                                force_paging=False,
                                dynamic_gm=dynamic_game,
                                send_obs=True,
                                auto_minimize=auto_minimize,
                                send_gamestates=send_gamestate,
                                gamemode_weights=gamemode_weights,  # default 1/3
                                streamer_mode=streamer_mode,
                                deterministic_streamer=deterministic_streamer,
                                force_old_deterministic=force_old_deterministic,
                                # testing
                                batch_mode=batch_mode,
                                step_size=Constants_demo.STEP_SIZE,
                                # full_team_evaluations=True,
                                epic_rl_exe_path=epic_rl_exe_path,
                                simulator=simulator,
                                visualize=visualize,
                                live_progress=False,
                                tick_skip=Constants_demo.FRAME_SKIP,
                                pretrained_agents=pretrained_agents,
                                )

    worker.env._match._obs_builder.env = worker.env  # noqa
    if simulator and visualize:
        from rocketsimvisualizer import VisualizerThread

        arena = worker.env._game.arena  # noqa
        v = VisualizerThread(arena, fps=60, tick_rate=120, tick_skip=frame_skip, step_arena=False,  # noqa
                             overwrite_controls=False)  # noqa
        v.start()

    worker.run()
