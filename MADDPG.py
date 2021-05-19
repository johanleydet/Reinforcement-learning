import matplotlib

matplotlib.use("TkAgg")
import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim


import argparse
from itertools import count

import os, sys, random
from torch.utils.tensorboard import SummaryWriter
from utils import *



class Replay_buffer():
    # permet de stocker les transitions
    def __init__(self, max_size=1000):
        self.storage = []
        self.max_size = max_size
        self.pointer = 0

    def add(self, data):
        # si l'on est taille max remplacer une donnée puis icrementer le pointer modulo taille max
        if len(self.storage) == self.max_size:
            self.storage[self.pointer] = data
            self.pointer = (self.pointer + 1) % self.max_size
        # sinon ajouter une donnée
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        # tirer batch_size entiers
        indices = np.random.randint(0, len(self.storage), size=batch_size)
        # création des listes
        s, s_prime, a, r, d = [], [], [], [], []
        # remplir les listes
        for i in indices:
            S, S_PRIME, A, R, D = self.storage[i]
            s.append(S)
            s_prime.append(S_PRIME)
            a.append(A)
            r.append(R)
            d.append(D)
        return torch.FloatTensor(s), torch.FloatTensor(s_prime),torch.FloatTensor(a), torch.FloatTensor(r).unsqueeze(1), torch.FloatTensor(d).unsqueeze(1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class MADDPG(object):
    def __init__(self, STATE_DIM, ACTION_DIM, MAX_ACTION, MIN_ACTION,N_AGENTS):
        self.UPDATE_ITERATION = 200
        self.EXPLORATION_NOISE = 0.1
        self.RHO = 0.005
        self.BATCH_SIZE = 130
        self.GAMMA = 0.95
        self.N_AGENTS = N_AGENTS
        self.ACTION_DIM = ACTION_DIM
        self.ACTION_MIN = MIN_ACTION
        self.ACTION_MAX = MAX_ACTION

        self.replay_buffer = Replay_buffer()

        self.actor = []
        self.actor_target = []
        self.actor_optimizer = []
        self.critic=[]
        self.critic_target=[]
        self.critic_optimizer=[]
        for i in range(self.N_AGENTS):
            self.actor.append(Actor(STATE_DIM, ACTION_DIM, MAX_ACTION))
            self.actor_target.append(Actor(STATE_DIM, ACTION_DIM, MAX_ACTION))
            self.actor_target[i].load_state_dict(self.actor[i].state_dict())
            self.actor_optimizer.append(optim.Adam(self.actor[i].parameters(), lr=1e-3))

            self.critic.append(Critic(STATE_DIM*N_AGENTS, ACTION_DIM*N_AGENTS))
            self.critic_target.append(Critic(STATE_DIM*N_AGENTS, ACTION_DIM*N_AGENTS))
            self.critic_target[i].load_state_dict(self.critic[i].state_dict())
            self.critic_optimizer.append(optim.Adam(self.critic[i].parameters(), lr=1e-3))



    def act(self, state,agent):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor[agent](state).data.numpy().flatten()
        noisy_action = action + np.random.normal(0, self.EXPLORATION_NOISE, size=self.ACTION_DIM)
        noisy_action_clip=noisy_action.clip([self.ACTION_MIN,self.ACTION_MIN], [self.ACTION_MAX,self.ACTION_MAX])
        return noisy_action_clip

    def update(self):
        for agent in range(self.N_AGENTS):
            state, next_state, action, reward, done = self.replay_buffer.sample(self.BATCH_SIZE)

            # state shape : self.BATCH_SIZE x N_AGENTS x STATE_DIM
            # next_state shape : self.BATCH_SIZE x N_AGENTS x STATE_DIM
            # action shape : self.BATCH_SIZE x N_AGENTS x ACTION_DIM
            # reward shape : self.BATCH_SIZE x 1 x N_AGENTS
            # done shape : self.BATCH_SIZE x 1 x N_AGENTS

            # Calcul de Y
            mu_target=torch.zeros(self.BATCH_SIZE, self.N_AGENTS, self.ACTION_DIM)
            for i in range(self.N_AGENTS):
                mu_target[:,i,:] = self.actor_target[i](next_state[:,i,:])

            Q_target = self.critic_target[agent](next_state.view(self.BATCH_SIZE,-1), mu_target.view(self.BATCH_SIZE,-1))
            Y = (reward[:,:,agent] + self.GAMMA * Q_target).detach()

            # Q courant
            current_Q = self.critic[agent](state.view(self.BATCH_SIZE,-1), action.view(self.BATCH_SIZE,-1))

            # 13 :Optimisation de la critique (phi)
            critic_loss = F.mse_loss(current_Q, Y)
            self.critic_optimizer[agent].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[agent].step()

            # 14 : Optimisation de l'acteur (theta)
            # on met un moins car c'est une montée de gradient

            action[:, agent, :] = self.actor[agent](state[:, agent, :])
            actor_loss = -self.critic[agent](state.view(self.BATCH_SIZE,-1), action.view(self.BATCH_SIZE,-1)).mean()
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[agent].step()

            # 15 : maj soft des paramètres des réseaux cibles
        for i in range(self.N_AGENTS):
            for param, target_param in zip(self.critic[i].parameters(), self.critic_target[i].parameters()):
                target_param.data.copy_(self.RHO * param.data + (1 - self.RHO) * target_param.data)

            for param, target_param in zip(self.actor[i].parameters(), self.actor_target[i].parameters()):
                target_param.data.copy_(self.RHO * param.data + (1 - self.RHO) * target_param.data)


    def save(self):
        pass

    def load(self):
        pass





"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world

if __name__ == '__main__':
    logger = LogMe(SummaryWriter("./XP/"))
    # action_N shape : N_agent x 2
    # action_N type : list numpy.ndarray numpy.float64

    # next_state_N shape : N_agent x 14
    # next_state_N type : list numpy.ndarray numpy.float64

    # reward_N shape : N_agent
    # reward_N type : list numpy.float64

    # done_N shape : N_agent
    # done_N type : list de bool

    N_AGENTS=3
    STATE_DIM=14
    ACTION_DIM=2
    MAX_ACTION=1
    MIN_ACTION=-1
    RENDER_INTERVAL = 50
    MAX_ITERATION=25
    EPISODES=40000

    agents = MADDPG(STATE_DIM, ACTION_DIM, MAX_ACTION, MIN_ACTION,N_AGENTS)
    env,scenario,world = make_env('simple_spread')



    for m in range(EPISODES):
        print("Episode : ", m)
        rsum=0
        state_N = env.reset()
        for k in range(MAX_ITERATION):
            # if k%RENDER_INTERVAL==0:
            #     env.render(mode="none")

            action_N=[]
            for agent in range(N_AGENTS):
                state=state_N[agent]
                action_N.append(agents.act(state,agent))


            next_state_N, reward_N, done_N, _ = env.step(action_N )

            agents.replay_buffer.add((state_N, next_state_N, action_N, reward_N, done_N))

            state_N = next_state_N

            agents.update()

            rsum+=np.mean(reward_N)

        logger.direct_write("rewards_mean",rsum, m)

    env.close()