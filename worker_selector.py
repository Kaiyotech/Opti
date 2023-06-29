import sys
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from rlgym.envs import Match
from rocket_learn.matchmaker.matchmaker import Matchmaker
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.state_setters.default_state import DefaultState

from CoyoteObs import CoyoteObsBuilder
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, \
    NoTouchTimeoutCondition
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from rocket_learn.utils.truncated_condition import TerminalToTruncatedWrapper
from CoyoteParser import SelectorParser
from rewards import ZeroSumReward
from torch import set_num_threads
from selection_listener import SelectionListener
from setter import CoyoteSetter
from mybots_statesets import EndKickoff, HalfFlip
from mybots_terminals import RandomTruncationBallGround
import Constants_selector
import numpy as np
import collections
import threading
import json
import os

from pretrained_agents.necto.necto_v1 import NectoV1
from pretrained_agents.nexto.nexto_v2 import NextoV2
from pretrained_agents.KBB.kbb import KBB
from pretrained_agents.GP.GP import GP

from rlgym.utils.common_values import BALL_RADIUS, BACK_WALL_Y

set_num_threads(1)


class ObsInfo:
    """keeps track of duplicate obs information"""

    def __init__(self, tick_skip, selector_infinite_boost: dict, dtap_dict) -> None:
        from rlgym.utils.common_values import BOOST_LOCATIONS
        self.boost_locations = np.array(BOOST_LOCATIONS)
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.inverted_boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.blue_obs = None
        self.orange_obs = None
        self.demo_timers = None
        self.BOOST_TIMER_STD = 10
        self.DEMO_TIMER_STD = 3
        self.time_interval = tick_skip / 120
        self.dodge_deadzone = 0.8
        self.any_timers = True
        self.boosttimes = [0] * 8
        self.jumptimes = [0] * 8
        self.fliptimes = [0] * 8
        self.has_flippeds = [False] * 8
        self.has_doublejumpeds = [False] * 8
        self.flipdirs = [[0] * 2 for _ in range(8)]
        self.airtimes = [0] * 8
        self.on_grounds = [False] * 8
        self.prev_prev_actions = [[0] * 8 for _ in range(8)]
        self.is_jumpings = [False] * 8
        self.has_jumpeds = [False] * 8
        self.handbrakes = [0] * 8
        self.selector_infinite_boost = selector_infinite_boost
        self.floor_bounce = False
        self.backboard_bounce = False
        self.prev_ball_vel = np.asarray([0] * 3)
        self.dtap_dict = dtap_dict
        self.n = 0

    def reset(self, initial_state: GameState):
        self.n = 0
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.demo_timers = np.zeros(max(p.car_id for p in initial_state.players) + 1)
        self.blue_obs = []
        self.orange_obs = []

        # timers
        self.boosttimes = np.zeros(8)
        #
        # self.jumptimes = np.zeros(
        #     max(p.car_id for p in initial_state.players) + 1)

        for i in range(len(initial_state.players) + 1):
            if self.has_flippeds[i]:
                self.fliptimes[i] = 78
        self.has_flippeds = [False] * 8
        # self.has_doublejumpeds = [False] * (max(p.car_id for p in initial_state.players) + 1)
        # self.flipdirs = [[0] * 2 for _ in range(max(p.car_id for p in initial_state.players) + 1)]

        # self.airtimes = np.zeros(
        #     max(p.car_id for p in initial_state.players) + 1)

        self.prev_prev_actions = [[0] * 8 for _ in range(max(p.car_id for p in initial_state.players) + 1)]
        self.is_jumpings = [False] * 8
        # self.has_jumpeds = [False] * (max(p.car_id for p in initial_state.players) + 1)
        self.on_grounds = [False] * 8
        for p in initial_state.players:
            self.on_grounds[p.car_id] = p.on_ground

        # self.handbrakes = np.zeros(
        #     max(p.car_id for p in initial_state.players) + 1)

        self.floor_bounce = False
        self.backboard_bounce = False
        self.prev_ball_vel = np.array(initial_state.ball.linear_velocity)

    def pre_step(self, state: GameState):
        self.n = 0
        # create player/team agnostic items (do these even exist?)
        self._update_timers(state)
        # create team specific things
        self.blue_obs = self.boost_timers / self.BOOST_TIMER_STD
        self.orange_obs = self.inverted_boost_timers / self.BOOST_TIMER_STD
        inf_boost = self.selector_infinite_boost["infinite_boost"]
        if inf_boost:
            for player in state.players:
                player.boost_amount = 1
        else:
            for player in state.players:
                player.boost_amount /= 1

        # for double tap
        touched = False
        for player in state.players:
            if player.ball_touched:
                touched = True
        ball_bounced_ground = self.prev_ball_vel[2] * state.ball.linear_velocity[2] < 0
        ball_near_ground = state.ball.position[2] < BALL_RADIUS * 2
        if not touched and ball_near_ground and ball_bounced_ground:
            self.floor_bounce = True

        ball_bounced_backboard = self.prev_ball_vel[1] * state.ball.linear_velocity[1] < 0
        ball_near_wall = abs(state.ball.position[1]) > (BACK_WALL_Y - BALL_RADIUS * 2)
        if not touched and ball_near_wall and ball_bounced_backboard:
            self.backboard_bounce = True
            self.dtap_dict["ball_hit_bb"] = False

        if touched and not self.dtap_dict["hit_towards_bb"]:
            self.dtap_dict["hit_towards_bb"] = True

        if touched and self.dtap_dict["hit_towards_bb"] and self.dtap_dict["ball_hit_bb"]:
            self.dtap_dict["hit_towards_goal"] = True

        self.prev_ball_vel = np.array(state.ball.linear_velocity)

    def _update_timers(self, state: GameState):
        current_boosts = state.boost_pads
        boost_locs = self.boost_locations
        demo_states = [[p.car_id, p.is_demoed] for p in state.players]

        for i in range(len(current_boosts)):
            if current_boosts[i] == self.boosts_availability[i]:
                if self.boosts_availability[i] == 0:
                    self.boost_timers[i] = max(0, self.boost_timers[i] - self.time_interval)
            else:
                if self.boosts_availability[i] == 0:
                    self.boosts_availability[i] = 1
                    self.boost_timers[i] = 0
                else:
                    self.boosts_availability[i] = 0
                    if boost_locs[i][2] == 73:
                        self.boost_timers[i] = 10.0
                    else:
                        self.boost_timers[i] = 4.0
        self.boosts_availability = current_boosts
        self.inverted_boost_timers = self.boost_timers[::-1]
        self.inverted_boosts_availability = self.boosts_availability[::-1]

        for cid, dm in demo_states:
            if dm == True:  # Demoed
                prev_timer = self.demo_timers[cid]
                if prev_timer > 0:
                    self.demo_timers[cid] = max(0, prev_timer - self.time_interval)
                else:
                    self.demo_timers[cid] = 3
            else:  # Not demoed
                self.demo_timers[cid] = 0

    def step(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        self._update_addl_timers(player, state, previous_action)
        self.prev_prev_actions[player.car_id] = previous_action  # noqa

    # def after_step(self):
    #     self.n += 1

    def _update_addl_timers(self, player: PlayerData, state: GameState, prev_actions: np.ndarray):
        cid = player.car_id

        # if this player was not boosting last tick and their boosttime timer means they actually stopped boosting, set to 0
        if prev_actions[6] == 0 and self.boosttimes[cid] == 12:
            self.boosttimes[cid] = 0
        # otherwise, just increment the boosttime
        else:
            self.boosttimes[cid] += self.time_interval * 120
            self.boosttimes[cid] = min(12, self.boosttimes[cid])

        # update jumptime
        if self.on_grounds[cid] and not self.is_jumpings[cid]:
            self.has_jumpeds[cid] = False

        if self.is_jumpings[cid]:
            # JUMP_MIN_TIME = 3 ticks
            # JUMP_MAX_TIME = 24 ticks
            # if not ((self.jumptimes[cid] < 3 or prev_actions[5] == 1) and self.jumptimes[cid] < 24):
            #     self.is_jumpings[cid] = self.jumptimes[cid] < 3
            self.is_jumpings[cid] = self.jumptimes[cid] < 3 or (prev_actions[5] == 1 and self.jumptimes[cid] < 24)
        elif prev_actions[5] == 1 and self.prev_prev_actions[cid][5] == 0 and self.on_grounds[cid]:
            self.is_jumpings[cid] = True
            self.jumptimes[cid] = 0

        if self.is_jumpings[cid]:
            self.has_jumpeds[cid] = True
            self.jumptimes[cid] += self.time_interval * 120
            self.jumptimes[cid] = min(
                24, self.jumptimes[cid])
        else:
            self.jumptimes[cid] = 0

        # update airtime and fliptime
        if player.on_ground:
            self.has_doublejumpeds[cid] = False
            self.has_flippeds[cid] = False
            self.airtimes[cid] = 0
            self.fliptimes[cid] = 0
            self.flipdirs[cid] = [0, 0]
            self.on_grounds[cid] = True
        else:
            if self.has_jumpeds[cid] and not self.is_jumpings[cid]:
                self.airtimes[cid] += self.time_interval * 120
                # DOUBLEJUMP_MAX_DELAY = 150 ticks
                self.airtimes[cid] = min(
                    150, self.airtimes[cid])
            else:
                self.airtimes[cid] = 0
            if self.has_jumpeds[cid] and (prev_actions[5] == 1 and self.prev_prev_actions[cid][5] == 0) and \
                    self.airtimes[cid] < 150:
                if not self.has_doublejumpeds[cid] and not self.has_flippeds[cid]:
                    should_flip = max(max(abs(prev_actions[3]), abs(prev_actions[2])), abs(
                        prev_actions[4])) >= self.dodge_deadzone
                    if should_flip:
                        self.fliptimes[cid] = 0
                        self.has_flippeds[cid] = True
                        flipdir = np.asarray(
                            [-prev_actions[2], prev_actions[3] + prev_actions[4]])
                        if np.any(flipdir):
                            self.flipdirs[cid] = list(
                                flipdir / np.linalg.norm(flipdir))
                        else:
                            self.flipdirs[cid] = [0, 0]
                    else:
                        self.has_doublejumpeds[cid] = True
        if self.has_flippeds[cid]:
            self.fliptimes[cid] += self.time_interval * 120
            # FLIP_TORQUE_TIME = 78 ticks
            self.fliptimes[cid] = min(
                78, self.fliptimes[cid])

        # update handbrake
        if prev_actions[7] == 1:
            # POWERSLIDE_RISE_RATE = 5
            self.handbrakes[cid] += 5 * self.time_interval
            self.handbrakes[cid] = min(
                1, self.handbrakes[cid])
        else:
            # POWERSLIDE_FALL_RATE = 2
            self.handbrakes[cid] -= 2 * self.time_interval
            self.handbrakes[cid] = max(
                0, self.handbrakes[cid])


class SelectionDispatcher(SelectionListener):
    """Dispatches model selection messages to redis channel"""

    def __init__(self, redis, redis_channel) -> None:
        super().__init__()
        self.redis = redis
        self.redis_channel = redis_channel
        self.xthread_queue = collections.deque()
        self.wake_event = threading.Event()
        self.should_run = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        # delete all stats on startup
        for key in r.scan_iter("selector_stat*"):
            r.delete(key)

    def _flush_queue(self):
        if len(self.xthread_queue) == 0:
            return

        pipe = self.redis.pipeline()
        while len(self.xthread_queue) > 0:
            selected_model_name, model_action = self.xthread_queue.popleft()
            selection_message = dict(model=selected_model_name, actions=model_action.tolist())
            selection_message = json.dumps(selection_message)
            pipe.publish(self.redis_channel, selection_message)
        pipe.execute()
        self.wake_event.clear()

    def _run(self):
        while self.should_run:
            self.wake_event.wait()
            self._flush_queue()

    def on_selection(self, selected_model_name: str, model_action: np.ndarray):
        self.xthread_queue.append((selected_model_name, model_action))
        self.wake_event.set()

    def stop(self):  # unused
        self.xthread_queue.clear()
        self.should_run = False
        self.wake_event.set()
        self.thread.join()


if __name__ == "__main__":
    frame_skip = Constants_selector.FRAME_SKIP
    dtap_status = {"hit_towards_bb": False,
                   "ball_hit_bb": False,
                   "hit_towards_goal": False,
                   }
    rew = ZeroSumReward(zero_sum=Constants_selector.ZERO_SUM,
                        tick_skip=frame_skip,
                        goal_w=10,
                        concede_w=-10,
                        team_spirit=1,
                        demo_w=3,
                        got_demoed_w=-3,
                        punish_action_change_w=0,
                        decay_punish_action_change_w=0,
                        flip_reset_w=0.5,
                        flip_reset_goal_w=3,
                        aerial_goal_w=3,
                        double_tap_w=4,
                        # cons_air_touches_w=,
                        # jump_touch_w=0.5,
                        wall_touch_w=2,
                        flatten_wall_height=True,
                        # exit_velocity_w=1,
                        # acel_ball_w=1,
                        # backboard_bounce_rew=2,
                        velocity_pb_w=0,  # 0.005,
                        velocity_bg_w=0.02,
                        kickoff_w=0.05,
                        punish_dist_goal_score_w=-1,
                        # # boost_gain_w=0.15,
                        # # punish_boost=False,
                        # # use_boost_punish_formula=False,
                        # # boost_spend_w=0,  # -0.1,
                        # # boost_gain_small_w=0.15,
                        # # punish_low_boost_w=-0.02,
                        # cancel_jump_touch_indices=[0, 1, 2, 4, 5, 9, *range(10, 28)],
                        # cancel_wall_touch_indices=[0, 1, 2, 4, 5, 9, *range(10, 28)],
                        cancel_flip_reset_indices=[0, 1, 2, 4, 5, 9, *range(10, 28)],
                        # cancel_cons_air_touch_indices=[0, 1, 2, 4, 5, 9, *range(10, 28)],
                        # cancel_backboard_bounce_indices=[0, 1, 2, 4, 5, 9, *range(10, 28)],
                        dtap_dict=dtap_status,
                        aerial_reward_w=0.01,
                        ground_reward_w=0.001,
                        defend_reward_w=0.001,
                        wall_reward_w=0.015,
                        aerial_indices=[3, 6, 7, 8, 28, 29],
                        wall_indices=[8, 25, 26, 28, 29],
                        ground_indices=[0, 1, 2, 4, 5, *range(9, 25), 27, 29],
                        defend_indices=[3, 6, 7, 8, 28],
                        )
    # obs_output = np.zeros()

    # simple_actions = [32, 33, 34, 35, 36, 37]

    selector_infinite_boost = {"infinite_boost": False}
    obs_info = ObsInfo(tick_skip=Constants_selector.FRAME_SKIP, selector_infinite_boost=selector_infinite_boost,
                       dtap_dict=dtap_status)
    parser = SelectorParser(obs_info=obs_info)
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    streamer_mode = False
    local = True
    auto_minimize = True
    game_speed = 100
    evaluation_prob = 0.01
    past_version_prob = 0.2
    deterministic_streamer = False
    force_old_deterministic = False
    team_size = 3
    dynamic_game = True
    infinite_boost_odds = 0.2
    host = "127.0.0.1"
    non_latest_version_prob = [0.8, 0.075, 0.075, 0.05]
    # non_latest_version_prob = [1, 0, 0, 0]
    gamemode_weights = {'1v1': 0.30, '2v2': 0.25, '3v3': 0.45}  # TODO testing fix this
    # gamemode_weights = {'1v1': 1, '2v2': 0, '3v3': 0}
    simulator = True
    visualize = False
    batch_mode = True

    model_name = "necto-model-30Y.pt"
    necto = NectoV1(model_string=model_name, n_players=6)
    model_name = "nexto-model.pt"
    nexto = NextoV2(model_string=model_name, n_players=6)
    model_name = "kbb.pt"
    kbb = KBB(model_string=model_name)
    model_name = "gp_jit.pt"
    gp = GP(model_string=model_name)

    pretrained_agents = Constants_selector.pretrained_agents
    # pretrained_agents = None

    matchmaker = Matchmaker(sigma_target=0.5, pretrained_agents=pretrained_agents, past_version_prob=past_version_prob,
                            full_team_trainings=0.8, full_team_evaluations=1, force_non_latest_orange=False,
                            non_latest_version_prob=non_latest_version_prob)

    terminals = [GoalScoredCondition(),
                 TerminalToTruncatedWrapper(RandomTruncationBallGround(avg_frames_per_mode=[fps * 20, fps * 30, fps * 40],
                                                             avg_frames=None,
                                                             min_frames=fps * 10)),
                 # TimeoutCondition(fps * 15),
                 # NoTouchTimeoutCondition(fps * 30),
                 ]

    if len(sys.argv) > 1:
        host = sys.argv[1]
        if host != "127.0.0.1" and host != "localhost":
            local = False

    if len(sys.argv) > 2:
        name = sys.argv[2]

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  db=Constants_selector.DB_NUM,
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=20, base=1.5), 25),
                  db=Constants_selector.DB_NUM,
                  )


    def setup_streamer():
        global game_speed, evaluation_prob, past_version_prob, auto_minimize, infinite_boost_odds, streamer_mode, \
            simulator, past_version_prob, pretrained_agents, non_latest_version_prob, matchmaker, terminals
        streamer_mode = True
        evaluation_prob = 0
        game_speed = 1
        auto_minimize = False
        infinite_boost_odds = 0
        simulator = False
        past_version_prob = 0
        dispatcher = SelectionDispatcher(r, Constants_selector.SELECTION_CHANNEL)
        parser.register_selection_listener(dispatcher)
        # terminals = [GoalScoredCondition(),
        #              TimeoutCondition(fps * 60),
        #              NoTouchTimeoutCondition(fps * 30),
        #              ]

        # pretrained_agents = {
        #     nexto: {'prob': 1, 'eval': True, 'p_deterministic_training': 1., 'key': "Nexto"},
        #     kbb: {'prob': 0, 'eval': True, 'p_deterministic_training': 1., 'key': "KBB"}
        # }

        non_latest_version_prob = [0, 1, 0, 0]

        matchmaker = Matchmaker(sigma_target=1, pretrained_agents=pretrained_agents,
                                past_version_prob=past_version_prob,
                                full_team_trainings=1, full_team_evaluations=1,
                                force_non_latest_orange=streamer_mode,
                                non_latest_version_prob=non_latest_version_prob,
                                showmatch=True,
                                orange_agent_text_file='orange_stream_file.txt'
                                )


    if len(sys.argv) > 3:
        if sys.argv[3] == 'GAMESTATE':
            send_gamestate = True
        elif sys.argv[3] == 'STREAMER':
            setup_streamer()
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

    obs_builder = CoyoteObsBuilder(expanding=True, tick_skip=Constants_selector.FRAME_SKIP, team_size=team_size,
                                   extra_boost_info=True, embed_players=True,
                                   stack_size=Constants_selector.STACK_SIZE,
                                   action_parser=parser, infinite_boost_odds=infinite_boost_odds, selector=True,
                                   selector_infinite_boost=selector_infinite_boost,
                                   doubletap_indicator=True,
                                   dtap_dict=dtap_status,
                                   flip_reset_counter=True,
                                   )
    # TODO fix testing
    setter = CoyoteSetter(mode="selector", dtap_dict=dtap_status)
    # setter = CoyoteSetter(mode="test_mirror", dtap_dict=dtap_status)
    # setter = HalfFlip()

    match = Match(
        game_speed=game_speed,
        spawn_opponents=True,
        team_size=team_size,
        state_setter=setter,
        obs_builder=obs_builder,
        action_parser=parser,
        terminal_conditions=terminals,
        reward_function=rew,
        tick_skip=frame_skip,
    ) if not simulator else Sim_Match(
        spawn_opponents=True,
        team_size=team_size,
        state_setter=setter,
        obs_builder=obs_builder,
        action_parser=parser,
        terminal_conditions=terminals,
        reward_function=rew,
    )

    #
    # pretrained_agents = {nectov1: 0.02, nexto: 0.02, kbb: 0.02, gp: 0}
    # # pretrained_agents = {nectov1: 0.1, nexto: 0.1, kbb: 0.1, gp: 0.1}

    worker = RedisRolloutWorker(r, name, match,
                                matchmaker=matchmaker,
                                sigma_target=2,
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
                                pretrained_agents=pretrained_agents,
                                # testing
                                eval_setter=EndKickoff(),
                                simulator=simulator,
                                live_progress=False,
                                visualize=visualize,
                                batch_mode=batch_mode,
                                step_size=Constants_selector.STEP_SIZE,
                                selector_skip_k=0.0007, # 1.6 seconds
                                # selector_boost_skip_k=0.0018,  # 1 seconds
                                # unlock_selector_indices=simple_actions,
                                # unlock_indices_group=simple_actions,
                                # parser_boost_split=parser.get_model_action_size(),
                                # initial_choice_block_indices=[2, 37],
                                # initial_choice_block_weight=0.5,
                                )

    worker.env._match._obs_builder.env = worker.env

    parser.force_selector_choice = worker.force_selector_choice  # ugh. I hate myself.

    if simulator and visualize:
        from rocketsimvisualizer import VisualizerThread

        arena = worker.env._game.arena  # noqa
        v = VisualizerThread(arena, fps=60, tick_rate=120, tick_skip=frame_skip, step_arena=False,  # noqa
                             overwrite_controls=False)  # noqa
        v.start()

    worker.run()
