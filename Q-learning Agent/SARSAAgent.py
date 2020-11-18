import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class SARSAAgent(object):
    def __init__(self, action_space, learning_rate=10e-4, discount_factor=.999, epsilon=.1):
        self.states, self.P = env.getMDP() # utiliser pr la conversion
        
        self.learning_rate=learning_rate
        self.discount_factor=discount_factor
        self.epsilon=epsilon
        self.nb_action=env.action_space.n
        self.nb_obs=env.observation_space.n
        
        self.Q=np.zeros((self.nb_obs,self.nb_action))
        
    def updateQ(self, obs, next_obs, reward, action): 
        next_obs=self.states[gridworld.GridworldEnv.state2str(next_obs)]
        
        next_action=self.epsilon_greedy(next_obs)
        
        Qsa=self.Q[obs,action]
        
        lr=self.learning_rate
        discount=self.discount_factor
        
        self.Q[obs, action] += lr * ( reward + discount * self.Q[next_obs, next_action] - Qsa )
    
    def epsilon_greedy(self, obs) :
        if np.random.random() < 1 - self.epsilon: 
            action=np.argmax(self.Q[obs, :])
        else : 
            action=np.random.randint(self.nb_action)
            
        return action
    
    def act(self, obs, reward, done):
        obs=self.states[gridworld.GridworldEnv.state2str(obs)]
        action=self.epsilon_greedy(obs)
        
        return action, obs
        
    

if __name__ == '__main__':
    
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = SARSAAgent(env.action_space)


    episode_count = 2000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    all_rsum = []
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 2000 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        #rsum = 0
        while True:
            action, obs = agent.act(obs, reward, done)
            
            old_obs=obs
            
            obs, reward, done, _ = env.step(action)
            
            agent.updateQ(old_obs, obs, reward, action)
            
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
        all_rsum.append(rsum)
    print("done")
    all_rsum = np.array(all_rsum)

    
    env.close()
    
    plt.title("Reward cumulé")
    plt.xlabel("iterations")
    plt.ylabel("reward cumulé")
    plt.plot(all_rsum, label="SARSA_Q_learning")
    plt.legend()
    plt.show()