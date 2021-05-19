import argparse
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from utils import *

class Replay_buffer():
    # permet de stocker les transitions
    def __init__(self, max_size=100000):
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

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()

        self.UPDATE_ITERATION=200
        self.EXPLORATION_NOISE = 0.1
        self.RHO=0.005
        self.BATCH_SIZE=100
        self.GAMMA=0.5

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).data.numpy().flatten()
        noisy_action = action + np.random.normal(0, self.EXPLORATION_NOISE, size=env.action_space.shape[0])
        noisy_action_clip=noisy_action.clip(env.action_space.low, env.action_space.high)

        # print(env.action_space.high)

        return noisy_action_clip

    def update(self):

        for it in range(self.UPDATE_ITERATION):
            # 11 : sample
            state, next_state, action, reward, done = self.replay_buffer.sample(self.BATCH_SIZE)

            # Calcul de Y
            mu_target = self.actor_target(next_state)
            Q_target = self.critic_target(next_state, mu_target)
            Y = reward + ((1-done) * self.GAMMA * Q_target).detach()

            # Q courant
            current_Q = self.critic(state, action)

            # 13 :Optimisation de la critique (phi)
            critic_loss = F.mse_loss(current_Q, Y)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 14 : Optimisation de l'acteur (theta)
            # on met un moins car c'est une montée de gradient
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 15 : maj soft des paramètres des réseaux cibles
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.RHO * param.data + (1 - self.RHO) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.RHO * param.data + (1 - self.RHO) * target_param.data)


    def save(self):
        pass

    def load(self):
        pass


if __name__ == '__main__':

    outdir = "./XP/"
    logger = LogMe(SummaryWriter(outdir))

    # env = gym.make("Pendulum-v0")
    # env = gym.make("LunarLanderContinuous-v2")
    env = gym.make("MountainCarContinuous-v0")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # env.seed(0)
    # torch.manual_seed(0)
    # np.random.seed(0)

    agent = DDPG(state_dim, action_dim, max_action)

    RENDER_INTERVAL = 50
    MAX_EPISODE = 10000

    for i in range(MAX_EPISODE):
        rsum = 0
        step = 0

        state = env.reset()
        for t in count():
            if i % RENDER_INTERVAL == 0:
                env.render()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.add((state, next_state, action, reward, np.float(done)))

            state=next_state

            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(t+1) + " actions")
                break
            step += 1
            rsum += reward

        logger.direct_write("reward", rsum, i)
        agent.update()
    env.close()
