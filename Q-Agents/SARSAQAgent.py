import matplotlib.pyplot as plt
import gym
import gridworld
import numpy as np

class SARSAAgent(object):
    def __init__(self, env, learning_rate=10e-4, discount_factor=.999, epsilon=.1):
        self.Q = {}
        self.env = env
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.last_state = None
        self.last_action = None


    def action(self, state, reward):
        state = self.env.state2str(state)

        if state not in self.Q:
            self.Q[state] = {}
            for i in range(self.env.action_space.n):
                self.Q[state][i]= 0

        action = self.epsilon_greedy(state)

        self.updateQ(action, state, reward)
        return action

    def epsilon_greedy(self, state):
        if np.random.random() < 1 - self.epsilon:
            action = max(self.Q[state], key=self.Q[state].get)
        else:
            action = np.random.randint(self.env.action_space.n)

        return action

    def updateQ(self, action, state, reward):
        if self.last_state == None and self.last_action == None:
            self.last_state = state
            self.last_action = action

        else:
            Qsa_prime = self.Q[state][action]
            Qsa = self.Q[self.last_state][self.last_action]

            lr = self.learning_rate
            discount = self.discount_factor

            self.Q[self.last_state][self.last_action] += lr * (reward + discount * Qsa_prime - Qsa)

            self.last_state = state
            self.last_action = action

if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human")  # visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ", len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}
    # Execution avec un Agent
    agent = SARSAAgent(env)
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

        while True:
            action = agent.action(obs, reward)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1

            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
        all_rsum.append(rsum)

    print("done")

    env.close()

    plt.title("Reward cumulé SARSAQ-learning")
    plt.xlabel("iterations")
    plt.ylabel("reward cumulé")
    plt.plot(all_rsum, label="SARSAQ-learning")
    plt.legend()
    plt.show()