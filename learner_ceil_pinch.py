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
import Constants_ceil_pinch

from utils.misc import count_parameters

import os
from torch import set_num_threads
from rocket_learn.utils.stat_trackers.common_trackers import Speed, Demos, TimeoutRate, Touch, EpisodeLength, Boost, \
    BehindBall, TouchHeight, DistToBall, AirTouch, AirTouchHeight, BallHeight, BallSpeed, CarOnGround, GoalSpeed,\
    MaxGoalSpeed
from my_stattrackers import GoalSpeedTop5perc

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
    frame_skip = Constants_ceil_pinch.FRAME_SKIP
    half_life_seconds = Constants_ceil_pinch.TIME_HORIZON
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    config = dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        n_steps=Constants_ceil_pinch.STEP_SIZE,
        batch_size=100_000,
        minibatch_size=50_000,
        epochs=30,
        gamma=gamma,
        save_every=10,
        model_every=100,
        ent_coef=0.01,
    )

    run_id = "ceil_pinch_run1.02"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="./wandb_store",
                        name="Ceil_Pinch_Run1.02",
                        project="Opti",
                        entity="kaiyotech",
                        id=run_id,
                        config=config,
                        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                        resume=True,
                        )
    redis = Redis(username="user1", password=os.environ["redis_user1_key"], db=Constants_ceil_pinch.DB_NUM)  # host="192.168.0.201",
    redis.delete("worker-ids")

    stat_trackers = [
        Speed(normalize=True), Demos(), TimeoutRate(), Touch(), EpisodeLength(), Boost(), BehindBall(), TouchHeight(),
        DistToBall(), AirTouch(), AirTouchHeight(), BallHeight(), BallSpeed(normalize=True), CarOnGround(),
        GoalSpeed(), MaxGoalSpeed(), GoalSpeedTop5perc(),
    ]

    rollout_gen = RedisRolloutGenerator("Opti_ceil_pinch",
                                        redis,
                                        lambda: CoyoteObsBuilder(expanding=True, tick_skip=Constants_ceil_pinch.FRAME_SKIP,
                                                                 team_size=3, extra_boost_info=False),
                                        lambda: ZeroSumReward(zero_sum=Constants_ceil_pinch.ZERO_SUM,
                                                              goal_w=0,
                                                              aerial_goal_w=10,
                                                              double_tap_w=20,
                                                              concede_w=-10,
                                                              velocity_pb_w=0.025,
                                                              velocity_bg_w=2,
                                                              acel_ball_w=2,
                                                              exit_velocity_w=3,
                                                              team_spirit=0,
                                                              cons_air_touches_w=3,
                                                              jump_touch_w=1,
                                                              wall_touch_w=0.5,
                                                              ),
                                        lambda: CoyoteAction(),
                                        save_every=logger.config.save_every * 3,
                                        model_every=logger.config.model_every,
                                        logger=logger,
                                        clear=False,
                                        stat_trackers=stat_trackers,
                                        # gamemodes=("1v1", "2v2", "3v3"),
                                        max_age=1,
                                        )

    critic = Sequential(Linear(222, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                        Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                        Linear(512, 1))

    actor = Sequential(Linear(222, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                       Linear(512, 373))

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

    alg.load("ceil_pinch_saves/Opti_1664986050.1984107/Opti_1020/checkpoint.pt")
    alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
    alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    alg.run(iterations_per_save=logger.config.save_every, save_dir="ceil_pinch_saves")
