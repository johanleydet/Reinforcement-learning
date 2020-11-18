import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy



class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.nbAction = action_space.n # renvoie 4 ie le nb d'action 
        self.states, self.P = env.getMDP()
        
    def valueFonction(s, pi):
        pass
        
    def fit(self, epsilon=.0001, gamma=.99):
        PI0=np.random.randint(self.nbAction,size=11)
        PI=[] # initialisation de PI
        
        
        PI.append(PI0)
        k=0
        
        
        B=True
        while(B): 
            V0=np.random.randint(self.nbAction,size=11)
            V=[] # initialisation de V
            
            V.append(V0)
            
            i=0
            
            b=True
            while(b):
              
                Vi=np.zeros(11) # un vecteur temporaire qui récoltera les Vi(s)
            
                # pour tout état s faire
                for state in self.P :
                    
                    # state la matrice représentant l'état et s sont numéro
                    s=self.states[state] 
                    action=PI[k][s] # nous donne l'action à réaliser
                   
                    sigma1=0
                    if action in self.P[state] : 
                    
                        # c'est le premier sigma de l'algorithme
                        for pb, destination, reward, done in self.P[state][action] :
                            sigma1+=pb*(reward+gamma*V[i][self.states[destination]])

                    Vi[s]=sigma1 # pour chaque état on à calulé v[s]
                V.append(Vi)  
                
                i+=1
                
                # condition d'arrêt de la boucle
                if np.linalg.norm(V[-1]-V[-2])<epsilon:
                    b=False
                    
            PIk=np.zeros(11)# un vecteur temporaire qui récoltera les PIk[s]
            
            # pour tout état s faire
            for state in self.P.keys() :
                # state la matrice représentant l'état et s sont numéro
                s=self.states[state]
                
                sigma2=[]# liste qui contiendra tous les sigmas_a 
                # ie sigma2 de l'algo selon a pour toute a
                # on s'es servira pour trouver l'argmax
                for action in range(self.nbAction):
                    
                    sigma2_a=0
                    if action in self.P[state] :
                        for pb, destination, reward, done in self.P[state][action] :
                            sigma2_a+=pb*(reward+gamma*V[i][self.states[destination]])

                    sigma2.append(sigma2_a)
                    
                PIk[s]=np.argmax(sigma2)
                                               
            PI.append(PIk)
            
            k+=1
                                               
            if list(PI[-1])==list(PI[-2]):
                B=False
                
        self.policy=PI[-1]
        

    def act(self, observation, reward, done):
        s=self.states[gridworld.GridworldEnv.state2str(observation)]
        return self.policy[s]


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
    agent = RandomAgent(env.action_space)
    agent.fit()

    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
