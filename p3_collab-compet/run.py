#!/usr/bin/env python
import argparse
from time import sleep
from unityagents import UnityEnvironment
from maddpg import MADDPGAgent
from config import config
from train import training_loop
import os
import math
import numpy as np
import torch
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', dest='train', help='Set the train mode')
parser.add_argument('--file_prefix', default=None, help='Set the file for agent to load weights with using prefix')
parser.add_argument('--playthroughs', default=10, type=int, help='Number of playthroughs played in a play mode')
arguments = parser.parse_args()

env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64', seed=config.general.seed)
brain_name = env.brain_names[0]
agent = MADDPGAgent(config=config, file_prefix=arguments.file_prefix)

if arguments.train:
    print('Train\n')
    training_loop(env, brain_name, agent, config)
else:
    print('Play\n')
    play_loop(env, brain_name, agent, playthrougs=arguments.playthroughs)


def play_loop(env, brain_name, agent, playthrougs=10):
    for e in range(playthrougs):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations
        done = False

        while not done:
            env_info = env.step(agent.act(state))[brain_name]
            state = env_info.vector_observations
            done = any(env_info.local_done)

def training_loop(env, brain_name, agent, config):
    scores = []
    avg_scores = []
    writer = SummaryWriter()
    last_max = -math.inf

    for e in range(1, config.training.episode_count):

        rewards = []
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        for t in range(config.training.max_t):
            rand = 0
            if e < 1000:
                rand = 1.0
            elif e < 2000:
                rand = 0.5
            action = agent.act(state, False, rand)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            if any(done):
                break

        scores.append(sum(np.array(rewards).sum(1)))
        avg_scores.append(sum(scores[-100:]) / 100)
        writer.add_scalar('stats/reward', scores[-1], e)
        writer.add_scalar('stats/avg_reward', avg_scores[-1], e)

        print(f'E: {e:6} | Average: {avg_scores[-1]:10.4f} | Best average: {max(avg_scores):10.4f}', end='\r')

        if e > 100 and avg_scores[-1] > last_max and ((avg_scores[-1] - last_max) > 0.05 or avg_scores[-1] > config.training.solve_score):
            for i, to_save in enumerate(agent.agents):
                torch.save(to_save.actor_local.state_dict(), os.getcwd() + f"/models/score_{avg_scores[-1]:.2f}_actor_{i}.weights")
                torch.save(to_save.critic_local.state_dict(), os.getcwd() + f"/models/score_{avg_scores[-1]:.2f}_critic_{i}.weights")
            last_max = avg_scores[-1]

        if e % 100 == 0:
            print(f'E: {e:6} | Average: {avg_scores[-1]:10.4f} | Best average: {max(avg_scores):10.4f}')

        if avg_scores[-1] > config.training.solve_score and not config.training.continue_after_solve:
            break