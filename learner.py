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
from Constants import FRAME_SKIP, TIME_HORIZON, ZERO_SUM

from utils.misc import count_parameters

import os
from torch import set_num_threads
from rocket_learn.utils.stat_trackers.common_trackers import Speed, Demos, TimeoutRate, Touch, EpisodeLength, Boost, \
    BehindBall, TouchHeight, DistToBall, AirTouch, AirTouchHeight, BallHeight, BallSpeed, CarOnGround, GoalSpeed,\
    MaxGoalSpeed
# TODO profile everything before starting to make sure everything is as fast as possible

# ideas for models:
# get to ball as fast as possible, sometimes with no boost, rewards exist
# pinches (ceiling and kuxir and team?), score in as few touches as possible with high velocity
# half flip, wavedash, wall dash, how to do this one?
# lix reset?
# normal play as well as possible, rewards exist
# aerial play without pinch, rewards exist
set_num_threads(1)

if __name__ == "__main__":
    frame_skip = FRAME_SKIP
    half_life_seconds = TIME_HORIZON
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    config = dict(
        actor_lr=2e-4,
        critic_lr=2e-4,
        n_steps=100_000,
        batch_size=100_000,
        minibatch_size=50_000,
        epochs=50,
        gamma=gamma,
        save_every=100,
        model_every=1000,
        ent_coef=0.01,
    )

    run_id = "test1"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="./wandb_store",
                        name="cAIyote",
                        project="cAIyoteV1",
                        entity="kaiyotech",
                        id=run_id,
                        config=config,
                        )
    redis = Redis(username="user1", password=os.environ["redis_user1_key"], db=1)  # host="192.168.0.201",
    redis.delete("worker-ids")

    stat_trackers = [
        Speed(normalize=True), Demos(), TimeoutRate(), Touch(), EpisodeLength(), Boost(), BehindBall(), TouchHeight(),
        DistToBall(), AirTouch(), AirTouchHeight(), BallHeight(), BallSpeed(normalize=True), CarOnGround(),
        GoalSpeed(), MaxGoalSpeed(),
    ]

    rollout_gen = RedisRolloutGenerator("cAIyote",
                                        redis,
                                        lambda: CoyoteObsBuilder(expanding=True, tick_skip=FRAME_SKIP, team_size=3),
                                        lambda: ZeroSumReward(zero_sum=ZERO_SUM),
                                        lambda: CoyoteAction(),
                                        save_every=logger.config.save_every,
                                        model_every=logger.config.model_every,
                                        logger=logger,
                                        clear=True, # TODO check this
                                        stat_trackers=stat_trackers,
                                        # gamemodes=("1v1", "2v2", "3v3"),
                                        max_age=1,
                                        )

    critic = Sequential(Linear(247, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),

                        Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512),
                        LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                        Linear(512, 1))

    actor = Sequential(Linear(247, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                       Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 91))

    actor = DiscretePolicy(actor, (91,))

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
    )

    # alg.load("model_saves/")
    alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
    alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    alg.run(iterations_per_save=logger.config.save_every, save_dir="model_saves")
