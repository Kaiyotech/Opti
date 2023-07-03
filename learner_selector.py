import wandb
import torch.jit

from torch.nn import Linear, Sequential, LeakyReLU

from redis import Redis
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from agent import OptiSelector, Opti
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from CoyoteObs import CoyoteObsBuilder

from CoyoteParser import SelectorParser
import numpy as np
from rewards import ZeroSumReward
import Constants_selector

from utils.misc import count_parameters

import math

import os
from torch import set_num_threads
from rocket_learn.utils.stat_trackers.common_trackers import Speed, Demos, TimeoutRate, Touch, EpisodeLength, Boost, \
    BehindBall, TouchHeight, DistToBall, AirTouch, AirTouchHeight, BallHeight, BallSpeed, CarOnGround, GoalSpeed, \
    MaxGoalSpeed
from my_stattrackers import GoalSpeedTop5perc, FlipReset, ActionGroupingTracker
from rlgym.utils.reward_functions.common_rewards import VelocityReward, EventReward
from rlgym.utils.reward_functions.combined_reward import CombinedReward

# ideas for models:
# get to ball as fast as possible, sometimes with no boost, rewards exist
# pinches (ceiling and kuxir and team?), score in as few touches as possible with high velocity
# half flip, wavedash, wall dash, how to do this one?
# lix reset?
# normal play as well as possible, rewards exist
# aerial play without pinch, rewards exist
# kickoff, 5 second terminal, reward ball distance into opp half
set_num_threads(1)

if __name__ == "__main__":
    frame_skip = Constants_selector.FRAME_SKIP
    half_life_seconds = Constants_selector.TIME_HORIZON
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    config = dict(
        actor_lr=1e-5,
        critic_lr=1e-5,
        n_steps=Constants_selector.STEP_SIZE,
        batch_size=100_000,
        minibatch_size=None,
        epochs=30,
        gamma=gamma,
        save_every=5,
        model_every=25,
        ent_coef=0.01,
    )

    run_id = "selector_run_23.00"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="./wandb_store",
                        name="Selector_Run_23.00",
                        project="Opti",
                        entity="kaiyotech",
                        id=run_id,
                        config=config,
                        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                        resume=True,
                        )
    redis = Redis(username="user1", password=os.environ["redis_user1_key"],
                  db=Constants_selector.DB_NUM)  # host="192.168.0.201",
    redis.delete("worker-ids")

    stat_trackers = [
        Speed(normalize=True), Demos(), TimeoutRate(), Touch(), EpisodeLength(), Boost(), BehindBall(), TouchHeight(),
        DistToBall(), AirTouch(), AirTouchHeight(), BallHeight(), BallSpeed(normalize=True), CarOnGround(),
        GoalSpeed(), MaxGoalSpeed(), GoalSpeedTop5perc(), FlipReset(),
    ]
    parser = SelectorParser()

    dtap_status = {"hit_towards_bb": False,
                   "ball_hit_bb": False,
                   "hit_towards_goal": False,
                   }
    initial_selector_skip_k = 0.0000475  # initial 6 seconds


    def skip_schedule(n_updates: int):
        if n_updates < 300:
            return initial_selector_skip_k / math.exp(-n_updates * (1 / 100))
        else:
            # step = initial_selector_skip_k / math.exp(-300 * (1 / 100))
            step = initial_selector_skip_k / math.exp(-300 * (1 / 300))
            # n_updates -= 665
            # 0.025 is about a quarter second, or 7.5 frames, seems a good minimum
            return min(step / math.exp(-n_updates * (1 / 300)), 0.025)


    rollout_gen = RedisRolloutGenerator("Opti_Selector",
                                        redis,
                                        lambda: CoyoteObsBuilder(expanding=True,
                                                                 tick_skip=Constants_selector.FRAME_SKIP,
                                                                 team_size=3, extra_boost_info=True,
                                                                 embed_players=True,
                                                                 stack_size=Constants_selector.STACK_SIZE,
                                                                 action_parser=parser,
                                                                 selector=True,
                                                                 doubletap_indicator=True,
                                                                 dtap_dict=dtap_status,
                                                                 flip_reset_counter=True,
                                                                 ),

                                        lambda: ZeroSumReward(zero_sum=Constants_selector.ZERO_SUM,
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
                                                              cancel_flip_reset_indices=[0, 1, 2, 4, 5, 9,
                                                                                         *range(10, 28)],
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
                                                              ),
                                        lambda: parser,
                                        save_every=logger.config.save_every * 3,
                                        model_every=logger.config.model_every,
                                        logger=logger,
                                        clear=False,
                                        stat_trackers=stat_trackers,
                                        # action_grouping_tracker=ActionGroupingTracker(aerial_indices=[3, 6, 7, 8, 28, 29],
                                        #                                               wall_indices=[8, 25, 26, 28, 29],
                                        #                                               ground_indices=[0, 1, 2, 4, 5, *range(9, 25), 27, 29],
                                        #                                               defend_indices=[3, 6, 7, 8, 28]),
                                        # gamemodes=("1v1", "2v2", "3v3"),
                                        max_age=1,
                                        pretrained_agents=Constants_selector.pretrained_agents,
                                        selector_skip_k=initial_selector_skip_k,
                                        selector_skip_schedule=skip_schedule,
                                        )
    action_size = 30
    # boost_size = 2
    input_size = 435 + (Constants_selector.STACK_SIZE * action_size)
    # shape = (action_size, boost_size)
    critic = Sequential(Linear(input_size, 256), LeakyReLU(), Linear(256, 256), LeakyReLU(),
                        Linear(256, 256), LeakyReLU(),
                        Linear(256, 1))

    actor = Sequential(Linear(input_size, 256), LeakyReLU(), Linear(256, 256), LeakyReLU(), Linear(256, 128),
                       LeakyReLU(),
                       Linear(128, action_size))

    critic = Opti(embedder=Sequential(Linear(35, 128), LeakyReLU(), Linear(128, 35 * 5)), net=critic,
                  )

    actor = Opti(embedder=Sequential(Linear(35, 128), LeakyReLU(), Linear(128, 35 * 5)), net=actor)

    actor = DiscretePolicy(actor, shape=(action_size,))

    actor.to("cuda")
    critic.to("cuda")

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": logger.config.actor_lr},
        {"params": critic.parameters(), "lr": logger.config.critic_lr},
    ], fused=True)

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    print(f"Gamma is: {gamma}")
    count_parameters(agent)

    action_dict = {i: k for i, k in enumerate(Constants_selector.SUB_MODEL_NAMES)}
    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=logger.config.ent_coef,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        logger=logger,
        zero_grads_with_none=True,
        disable_gradient_logging=True,
        action_selection_dict=action_dict,
        num_actions=action_size,
        # max_grad_norm=None,
    )

    alg.load("Selector_saves/Opti_1688274735.6054552/Opti_1075/checkpoint.pt")

    alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
    alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    # alg.freeze_policy(500)

    alg.run(iterations_per_save=logger.config.save_every, save_dir="Selector_saves")
