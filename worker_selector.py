import sys
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from rlgym.envs import Match
from rlgym.utils.gamestates import GameState

from CoyoteObs import CoyoteObsBuilder
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, \
    NoTouchTimeoutCondition
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from CoyoteParser import SelectorParser
from rewards import ZeroSumReward
from torch import set_num_threads
from selection_listener import SelectionListener
from setter import CoyoteSetter
import Constants_selector
import numpy as np
import collections
import threading
import json
import os

set_num_threads(1)


class ObsInfo:
    """keeps track of duplicate obs information"""
    def __init__(self, tick_skip) -> None:
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

    def reset(self, initial_state: GameState):
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.demo_timers = np.zeros(max(p.car_id for p in initial_state.players) + 1)
        self.blue_obs = []
        self.orange_obs = []

    def pre_step(self, state: GameState):
        # create player/team agnostic items (do these even exist?)
        self._update_timers(state)
        # create team specific things
        self.blue_obs = self.boost_timers / self.BOOST_TIMER_STD
        self.orange_obs = self.inverted_boost_timers / self.BOOST_TIMER_STD

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

    def stop(self): # unused
        self.xthread_queue.clear()
        self.should_run = False
        self.wake_event.set()
        self.thread.join()

if __name__ == "__main__":
    rew = ZeroSumReward(zero_sum=Constants_selector.ZERO_SUM,
                        goal_w=5,
                        concede_w=-5,
                        team_spirit=1,
                        punish_action_change_w=0,
                        decay_punish_action_change_w=-0.001,
                        flip_reset_w=2,
                        flip_reset_goal_w=5,
                        aerial_goal_w=2,
                        double_tap_w=4,
                        punish_directional_changes=True,
                        cons_air_touches_w=0.4,
                        jump_touch_w=0.5,
                        wall_touch_w=1,
                        exit_velocity_w=1,
                        velocity_pb_w=0.00,
                        kickoff_w=0.005,
                        )
    # obs_output = np.zeros()
    obs_info = ObsInfo(tick_skip=Constants_selector.FRAME_SKIP)
    parser = SelectorParser(obs_info=obs_info)
    frame_skip = Constants_selector.FRAME_SKIP
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    streamer_mode = False
    local = True
    auto_minimize = True
    game_speed = 100
    evaluation_prob = 0.01
    past_version_prob = 0.1
    deterministic_streamer = False
    force_old_deterministic = False
    team_size = 3
    dynamic_game = True
    infinite_boost_odds = 0
    host = "127.0.0.1"

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
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
                  db=Constants_selector.DB_NUM,
                  )

    def setup_streamer():
        global game_speed, evaluation_prob, past_version_prob, auto_minimize, infinite_boost_odds, streamer_mode
        streamer_mode = True
        evaluation_prob = 0
        game_speed = 1
        auto_minimize = False
        infinite_boost_odds = 0
        dispatcher = SelectionDispatcher(r, Constants_selector.SELECTION_CHANNEL)
        parser.register_selection_listener(dispatcher)

    if len(sys.argv) > 3:
        if sys.argv[3] == 'GAMESTATE':
            send_gamestate = True
        elif sys.argv[3] == 'STREAMER':
            setup_streamer()



    match = Match(
        game_speed=game_speed,
        spawn_opponents=True,
        team_size=team_size,
        state_setter=CoyoteSetter(mode="selector"),
        obs_builder=CoyoteObsBuilder(expanding=True, tick_skip=Constants_selector.FRAME_SKIP, team_size=team_size,
                                     extra_boost_info=True, embed_players=True,
                                     stack_size=Constants_selector.STACK_SIZE,
                                     action_parser=parser, infinite_boost_odds=infinite_boost_odds, selector=True,
                                     ),
        action_parser=parser,
        terminal_conditions=[GoalScoredCondition(),
                             NoTouchTimeoutCondition(fps * 40),
                             TimeoutCondition(fps * 300),
                             ],
        reward_function=rew,
        tick_skip=frame_skip,
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
                                gamemode_weights={'1v1': 0.15, '2v2': 0.2, '3v3': 0.65},  # default 1/3
                                streamer_mode=streamer_mode,
                                deterministic_streamer=deterministic_streamer,
                                force_old_deterministic=force_old_deterministic,
                                # testing
                                batch_mode=False,
                                step_size=Constants_selector.STEP_SIZE,
                                )

    worker.env._match._obs_builder.env = worker.env

    worker.run()
