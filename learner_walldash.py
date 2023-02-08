import wandb
import torch.jit

from torch.nn import Linear, Sequential, LeakyReLU

from redis import Redis
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from CoyoteObs import CoyoteObsBuilder

from CoyoteParser import CoyoteAction
import numpy as np
from rewards import ZeroSumReward
import Constants_walldash
from agent import MaskIndices

from utils.misc import count_parameters

import random

import os
from torch import set_num_threads
from rocket_learn.utils.stat_trackers.common_trackers import Speed, Demos, TimeoutRate, Touch, EpisodeLength, Boost, \
    BehindBall, TouchHeight, DistToBall, AirTouch, AirTouchHeight, BallHeight, BallSpeed, CarOnGround, GoalSpeed, \
    MaxGoalSpeed
from my_stattrackers import GoalSpeedTop5perc
from rlgym.utils.gamestates.physics_object import PhysicsObject

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
    frame_skip = Constants_walldash.FRAME_SKIP
    half_life_seconds = Constants_walldash.TIME_HORIZON
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    config = dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        n_steps=Constants_walldash.STEP_SIZE,
        batch_size=100_000,
        minibatch_size=None,
        epochs=30,
        gamma=gamma,
        save_every=10,
        model_every=1000,
        ent_coef=0.01,
    )

    run_id = "walldash_run5.00"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="./wandb_store",
                        name="Walldash_Run5.00",
                        project="Opti",
                        entity="kaiyotech",
                        id=run_id,
                        config=config,
                        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                        resume=True,
                        )
    redis = Redis(username="user1", password=os.environ["redis_user1_key"],
                  db=Constants_walldash.DB_NUM)  # host="192.168.0.201",
    redis.delete("worker-ids")

    stat_trackers = [
        Speed(normalize=True), EpisodeLength(), Boost(),
        DistToBall(), CarOnGround(),
    ]
    state = random.getstate()
    end_object = PhysicsObject()
    # end_object = None
    end_touched = {"Touched": False}
    rollout_gen = RedisRolloutGenerator("Walldash",
                                        redis,
                                        lambda: CoyoteObsBuilder(expanding=True,
                                                                 tick_skip=Constants_walldash.FRAME_SKIP,
                                                                 team_size=1, extra_boost_info=False,
                                                                 embed_players=False,
                                                                 add_jumptime=True,
                                                                 add_airtime=True,
                                                                 add_fliptime=True,
                                                                 add_boosttime=True,
                                                                 add_handbrake=True),
                                        lambda: ZeroSumReward(zero_sum=Constants_walldash.ZERO_SUM,
                                                              velocity_pb_w=0.02,
                                                              vp_end_object_w=0.02,
                                                              boost_gain_w=0.35,
                                                              boost_spend_w=4,
                                                              punish_boost=True,
                                                              touch_object_w=3.5,
                                                              touch_ball_w=3.5,
                                                              boost_remain_touch_w=2,
                                                              boost_remain_touch_object_w=2,
                                                              tick_skip=Constants_walldash.FRAME_SKIP,
                                                              walldash_w=1,
                                                              end_object=end_object,
                                                              end_touched=end_touched,
                                                              ),
                                        lambda: CoyoteAction(),
                                        save_every=logger.config.save_every * 3,
                                        model_every=logger.config.model_every,
                                        logger=logger,
                                        clear=False,
                                        stat_trackers=stat_trackers,
                                        gamemodes=("1v0",),
                                        max_age=1,
                                        )

    critic = Sequential(Linear(229, 256), LeakyReLU(), Linear(256, 256), LeakyReLU(),
                        Linear(256, 128), LeakyReLU(),
                        Linear(128, 1))

    actor = Sequential(Linear(229, 128), LeakyReLU(), Linear(128, 128), LeakyReLU(),
                       Linear(128, 128), LeakyReLU(),
                       Linear(128, 373))

    actor = DiscretePolicy(actor, (373,))

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": logger.config.actor_lr},
        {"params": critic.parameters(), "lr": logger.config.critic_lr}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    print(f"Gamma is: {gamma}")
    count_parameters(agent)

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

    )

    # alg.load("recovery_saves/Opti_1675358986.0893235/Opti_20/checkpoint.pt")
    alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
    alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    # alg.freeze_policy(20)

    alg.run(iterations_per_save=logger.config.save_every, save_dir="walldash_saves")
