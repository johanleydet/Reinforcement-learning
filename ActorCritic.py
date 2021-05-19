import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym
import warnings


import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings("ignore")

class ActorNetwork(nn.Module):
    '''
    NN actor
    '''
    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x

class ValueNetwork(nn.Module):
    '''
    NN critic
    '''
    def __init__(self,input_size,hidden_size,output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def target(rewards, gamma):
    '''
    calcul la véritable valeur de V
    '''
    N = len(rewards)
    weighted_rewards = gamma ** torch.arange(N) * rewards
    return torch.cumsum(weighted_rewards, dim=0)

def reverse_cumsum(x):
    '''
    somme cumulative de droite a gauche
    et non de gauche à droite qui serait siplement obtenue par cumsum
    '''
    return torch.flip(torch.cumsum(torch.flip(x, dims=(0,)), dim=0), dims=(0,))

def advantage(states, rewards,lambda_, gamma,NN):
    '''
    calcul la fonction avantage
    on choisit le lambda qui correspond au parametre de la difference temporelle
    si lambda est -1 c'est Monte Carlo
    si lambda est trop grand (par rapport au nombre d'action faites) on effectue aussi un Monte Carlo
    on renvoit donc deux parametres :
    - la valeur de la fonction avantage
    - un booleen pour specifier si on est en Monte Carlo
    '''
    N = len(rewards)
    # si lambda est trop grand ou en mode Monte Carlo ie -1
    if lambda_>=N or lambda_==-1:
        V = NN(states)
        weighted_rewards = gamma ** torch.arange(N) * rewards
        return (reverse_cumsum(weighted_rewards) - V,True)
    # sinon
    else :
        V = NN(states[:N - lambda_])
        V_prime=NN(states[lambda_:])
        L = rewards[:N - lambda_].shape[0]
        weighted_rewards = gamma ** torch.arange(L) * rewards[:N-lambda_]
        return (reverse_cumsum(weighted_rewards)+V_prime*gamma**L-V,False)

class A2C_agent(object):
    def __init__(self, env):
        self.ACTION_DIM=env.action_space.n # nb d'actions possibles
        self.STATE_DIM = env.observation_space.shape[0] # nb d'etat possible
        self.SAMPLE_NUMS = 1000 # nb de sample
        self.GAMMA=.99 # discount factor
        self.LAMBDA=-1 # difference temporelle -1 pour Monte Carlo

        # NN Value
        self.value_network = ValueNetwork(input_size=self.STATE_DIM, hidden_size=40, output_size=1)
        self.value_network_optim = torch.optim.Adam(self.value_network.parameters(), lr=0.01)
        # NN Actor
        self.actor_network = ActorNetwork(self.STATE_DIM, 40, self.ACTION_DIM)
        self.actor_network_optim = torch.optim.Adam(self.actor_network.parameters(), lr=0.01)

        self.loss = nn.SmoothL1Loss()

        # etat pour débuter la premiere itération du sample
        self.init_state=env.reset()

        # mode test ou train
        self.test=False

    def act(self,task, train=True, state=0):
        #mode train
        if train :
            # sample
            states, actions, rewards = self.sample(task)
            actions_var = torch.stack(actions).view(-1, self.ACTION_DIM)
            states_var = torch.Tensor(states).view(-1, self.STATE_DIM)
            rewards = torch.tensor(rewards)

            # train value network
            target_values = target(rewards, self.GAMMA)
            values = self.value_network(states_var)
            value_network_loss = self.loss(values.float(), target_values.float())
            value_network_loss.backward()
            self.value_network_optim.step()
            self.value_network_optim.zero_grad()

            # train actor network
            advantages,MC=advantage(states_var,rewards, self.LAMBDA, self.GAMMA,self.value_network)
            # important de faire la distinction entre MC et nn MC a cause des dimensions
            if not MC:
                N = len(rewards) # obliger car lambda peut etre nul
                log_softmax_actions = self.actor_network(states_var[:N-self.LAMBDA])
                actor_network_loss = - torch.mean(torch.sum(log_softmax_actions * actions_var[:N-self.LAMBDA], 1) * advantages)
            else :
                log_softmax_actions = self.actor_network(states_var)
                actor_network_loss = - torch.mean(torch.sum(log_softmax_actions * actions_var, 1) * advantages)

            actor_network_loss.backward()
            self.actor_network_optim.step()
            self.actor_network_optim.zero_grad()
        # mode test
        else :
            softmax_action = torch.exp(self.actor_network(torch.Tensor([state])))
            action = np.argmax(softmax_action.data.numpy()[0])
            return action

    def sample(self,task):
        states = []
        actions = []
        rewards = []

        state = self.init_state
        for j in range(self.SAMPLE_NUMS):
            # choisir une action avec la politique
            log_softmax_action = self.actor_network(torch.Tensor([state]))
            softmax_action = torch.exp(log_softmax_action)
            action = np.random.choice(self.ACTION_DIM, p=softmax_action.data.numpy()[0])

            # etape suivante
            next_state, reward, done, _ = task.step(action)

            # majs
            states.append(state)
            one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), self.ACTION_DIM)
            actions.append(one_hot_action)
            rewards.append(reward)
            state = next_state

            # si l'épisode est fini on reset l'état initial et on arrete la boucle for
            if done:
                self.init_state = task.reset()
                break
            else :
                self.init_state = state

        return states, actions, rewards

    def save(self,outputDir):
        pass

    def load(self,inputDir):
        pass


if __name__ == '__main__':
    # config = load_yaml('./configs/config_random_gridworld.yaml')
    # config = load_yaml('./configs/config_random_cartpole.yaml')
    config = load_yaml('./configs/config_random_lunar.yaml')

    # freqTest = config["freqTest"]
    freqTest = 10
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + "/random_" + "-" + tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()

    # agent = RandomAgent(env,config)
    agent=A2C_agent(env)

    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        if i % int(config["freqVerbose"]) == 0 and i >= config["freqVerbose"]:
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            # print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            # print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        if agent.test:

            while True:
                if verbose:
                    env.render()


                action = agent.act(env,False,ob)

                ob, reward, done, _ = env.step(action)
                j+=1

                rsum += reward
                if done:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    logger.direct_write("reward", rsum, i)
                    agent.nbEvents = 0
                    mean += rsum
                    rsum = 0
                    ob = env.reset()
                    break
        else:
            agent.act(env)
    env.close()
