import sys
from redis import Redis
from redis.retry import Retry  # noqa
from redis.backoff import ExponentialBackoff  # noqa
from redis.exceptions import ConnectionError, TimeoutError

from CoyoteObs import CoyoteObsBuilder
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from rocket_learn.matchmaker.matchmaker import Matchmaker
from rocket_learn.agent.types import PretrainedAgent
from rocket_learn.utils.truncated_condition import TerminalToTruncatedWrapper
from mybots_terminals import RandomTruncationBallGround
from CoyoteParser import CoyoteAction
from rewards import ZeroSumReward
from torch import set_num_threads
from setter import CoyoteSetter
from mybots_statesets import EndKickoff
import Constants_gp
import os

from pretrained_agents.necto.necto_v1 import NectoV1
from pretrained_agents.nexto.nexto_v2 import NextoV2
from pretrained_agents.KBB.kbb import KBB

from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, \
            NoTouchTimeoutCondition

set_num_threads(1)

if __name__ == "__main__":
    rew = ZeroSumReward(zero_sum=Constants_gp.ZERO_SUM,
                        goal_w=10,
                        concede_w=-10,
                        # double_tap_w=5,
                        velocity_bg_w=0.075 / 2,  # fix for the tick skip change
                        velocity_pb_w=0,
                        boost_gain_w=0,
                        punish_boost=True,
                        use_boost_punish_formula=False,
                        boost_spend_w=-0.45,
                        boost_gain_small_w=0,
                        punish_low_boost_w=-0.02,
                        demo_w=5,
                        acel_ball_w=1,
                        team_spirit=1,
                        # cons_air_touches_w=2,
                        jump_touch_w=6,
                        wall_touch_w=4,
                        touch_grass_w=0,
                        punish_bad_spacing_w=-0.1,
                        handbrake_ctrl_w=0,
                        tick_skip=Constants_gp.FRAME_SKIP,
                        flatten_wall_height=True,
                        slow_w=-0.1,
                        turtle_w=-0.2
                        )
    frame_skip = Constants_gp.FRAME_SKIP
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    streamer_mode = False
    local = True
    auto_minimize = True
    game_speed = 100
    evaluation_prob = 0.02
    past_version_prob = 1  # 0.5  # 0.1
    non_latest_version_prob = [0.825, 0.0826, 0.0578, 0.0346]  # this includes past_version and pretrained
    deterministic_streamer = True
    force_old_deterministic = True
    gamemode_weights = {'1v1': 0, '2v2': 1, '3v3': 0}
    visualize = False
    simulator = True
    batch_mode = True
    team_size = 3
    dynamic_game = True
    infinite_boost_odds = 0
    setter = CoyoteSetter(mode="normal", simulator=False)
    host = "127.0.0.1"
    epic_rl_exe_path = None  # "D:/Program Files/Epic Games/rocketleague_old/Binaries/Win64/RocketLeague.exe"

    model_name = "necto-model-30Y.pt"
    nectov1 = NectoV1(model_string=model_name, n_players=6)
    model_name = "nexto-model.pt"
    nexto = NextoV2(model_string=model_name, n_players=6)
    model_name = "kbb.pt"
    kbb = KBB(model_string=model_name)

    pretrained_agents = Constants_gp.pretrained_agents

    matchmaker = Matchmaker(sigma_target=0.5, pretrained_agents=pretrained_agents, past_version_prob=past_version_prob,
                            full_team_trainings=0.8, full_team_evaluations=1, force_non_latest_orange=False,
                            non_latest_version_prob=non_latest_version_prob)
                            
  
    terminals = [GoalScoredCondition(),
                 TerminalToTruncatedWrapper(
                     RandomTruncationBallGround(avg_frames_per_mode=[fps * 10, fps * 20, fps * 30],
                                                avg_frames=None,
                                                min_frames=fps * 10)),
                 ]

    if len(sys.argv) > 1:
        host = sys.argv[1]
        if host != "127.0.0.1" and host != "localhost":
            local = False
            batch_mode = True
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
            infinite_boost_odds = 0
            simulator = False
            past_version_prob = 0

            pretrained_agents = {
                nexto: {'prob': 1, 'eval': True, 'p_deterministic_training': 1., 'key': "Nexto"},
                kbb: {'prob': 0, 'eval': True, 'p_deterministic_training': 1., 'key': "KBB"}
            }

            non_latest_version_prob = [1, 0, 0, 0]

            matchmaker = Matchmaker(sigma_target=1, pretrained_agents=pretrained_agents,
                                    past_version_prob=past_version_prob,
                                    full_team_trainings=1, full_team_evaluations=1,
                                    force_non_latest_orange=streamer_mode,
                                    non_latest_version_prob=non_latest_version_prob,
                                    showmatch=True,
                                    orange_agent_text_file='orange_stream_file.txt'
                                    )
                                    
            gamemode_weights = {'1v1': 0.3, '2v2': 0.4, '3v3': 0.3}
            
            terminals = [GoalScoredCondition(), NoTouchTimeoutCondition(fps * 30), TimeoutCondition(fps * 300)]

            # setter = EndKickoff()

        elif sys.argv[3] == 'VISUALIZE':
            visualize = True

    if simulator:
        from rlgym_sim.envs import Match as Sim_Match
    else:
        from rlgym.envs import Match

    match = Match(
        game_speed=game_speed,
        spawn_opponents=True,
        team_size=team_size,
        state_setter=setter,
        obs_builder=CoyoteObsBuilder(expanding=True, tick_skip=Constants_gp.FRAME_SKIP, team_size=team_size,
                                     extra_boost_info=True, embed_players=True,
                                     infinite_boost_odds=infinite_boost_odds),
        action_parser=CoyoteAction(),
        terminal_conditions=terminals,
        reward_function=rew,
        tick_skip=frame_skip,
    ) if not simulator else Sim_Match(
        spawn_opponents=True,
        team_size=team_size,
        state_setter=setter,
        obs_builder=CoyoteObsBuilder(expanding=True, tick_skip=Constants_gp.FRAME_SKIP, team_size=team_size,
                                     extra_boost_info=True, embed_players=True,
                                     infinite_boost_odds=infinite_boost_odds),
        action_parser=CoyoteAction(),
        terminal_conditions=terminals,
        reward_function=rew,
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


    # pretrained_agents = {nectov1: 0, nexto: 0.05, kbb: 0.05}
    # pretrained_agents = {nexto: PretrainedAgent(prob=0.5, eval=True, p_deterministic_training=1., key="Nexto"),
    #                      kbb: PretrainedAgent(prob=0.5, eval=True, p_deterministic_training=1., key="KBB")}
    # pretrained_agents = None

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
                                step_size=Constants_gp.STEP_SIZE,
                                pretrained_agents=pretrained_agents,
                                eval_setter=EndKickoff(),
                                # full_team_evaluations=True,
                                epic_rl_exe_path=epic_rl_exe_path,
                                simulator=simulator,
                                visualize=False,
                                live_progress=False,
                                tick_skip=Constants_gp.FRAME_SKIP,
                                )

    worker.env._match._obs_builder.env = worker.env  # noqa
    if simulator and visualize:
        from rocketsimvisualizer import VisualizerThread
        arena = worker.env._game.arena  # noqa
        v = VisualizerThread(arena, fps=60, tick_rate=120, tick_skip=frame_skip, step_arena=False,  # noqa
                             overwrite_controls=False)  # noqa
        v.start()

    worker.run()
