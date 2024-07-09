import numpy as np
import matplotlib.pyplot as plt
import random

class MDPenvironment:
    def __init__(self, N):
        self.N = N
        self.S = self.generete_state_space()
        self.A = [a for a in range(N)]
        self.P = self.generate_markov_probabilities()
        self.r_stock = self.generate_stock_rewards()
        self.gamma = 0.9
        self.c = 0.01

    def reset(self):
        return random.randint(0,self.N-1)

    def get_binary_combinations(self):
        N = self.N
        B = {i: [] for i in range(2**N)}

        # Saving every combination of H/L given N
        for i in range(1 << N):
            i_b = format(i, '0' + str(N) + 'b')
            for j in range(N):
                B[i].append((1 if i_b[j] == '0' else 0))
        return B
    
    def generete_state_space(self):
        N = self.N
        S = {i: [] for i in range(N*np.power(2,N))}
        buffer = []
        B = self.get_binary_combinations()

        # State space creation

        # Getting all possible stocks we invest given the stock number
        for i in range(N):
            buffer.extend([i]*np.power(2,N))

        # Saving the these stocks as the first element of the list for each state
        for i in range(N*np.power(2,N)):
            S[i].append(buffer[i])

        # Getting all the different combinations of H/L for every stock played
        for i in range(N*np.power(2,N)):
            S[i].extend(B[i % np.power(2,N)])

        return S
    
    def generate_markov_probabilities(self):
        N = self.N
        P = np.zeros((N,4))

        if N == 2:        #pHH  pHL  pLH  pLL
            P = np.array([[0.7, 0.3, 0.3, 0.7],   #stock 0
                          [0.8, 0.2, 0.4, 0.6]])  #stock 1

        else:
            for i in range(N):
                if i< N/2 :
                    p_hl = 0.1
                    p_lh = 0.1

                else :
                    p_hl = 0.5
                    p_lh = 0.5

                P[i] = np.array([1-p_hl, p_hl, p_lh, 1-p_lh])

        return P.transpose()
    
    def generate_stock_rewards(self):
        N = self.N
        r = np.zeros((N,2))
        if N == 2:
            #              r_H    r_L
            r = np.array([[0.08, -0.04],
                          [0.04,  0.01]])
        else:
            for i in range(N):
                r[i][0] = np.random.uniform(-0.02, 0.1)
                r[i][1] = np.random.uniform(-0.02, 0.1)

        return r.transpose()
    
    def get_transition_probabilities(self, a, s):
        N = self.N
        states = self.S
        P = self.P

        s_prime = {i:states[k] for i,k in zip(range(np.power(2,N)),range(a*np.power(2,N),(a+1)*np.power(2,N)))}
        args_next = [k for k in range(a*np.power(2,N),(a+1)*np.power(2,N))]

        p = np.ones((np.power(2,N),))

        for i in range(np.power(2,N)):
            for n in range(N):
                if s[n+1] == 1 and s_prime[i][n+1] == 1:
                    p[i] *= P[0][n]
                elif s[n+1] == 1 and s_prime[i][n+1] == 0:
                    p[i] *= P[1][n]
                elif s[n+1] == 0 and s_prime[i][n+1] == 1:
                    p[i] *= P[2][n]
                else:
                    p[i] *= P[3][n]

        return p, args_next
    
    def get_transition_reward(self, a, s):
        r_stock = self.r_stock
        P = self.P
        c = self.c
        current_stock = s[0]
        current_state = s[a + 1]

        if a == current_stock:
            if current_state == 1:
                R = r_stock[0][a] * P[0][a] + r_stock[1][a] * P[1][a]
            else:
                R = r_stock[0][a] * P[2][a] + r_stock[1][a] * P[3][a]
        else:
            if current_state == 1:
                R = (r_stock[0][a] * P[0][a] + r_stock[1][a] * P[1][a]) - c
            else:
                R = (r_stock[0][a] * P[2][a] + r_stock[1][a] * P[3][a]) - c

        return R 
    

class Agent:
    def __init__(self, env):
        self.env = env

    def policy_evaluation(self,pi,epsilon):
        S = self.env.S
        A = self.env.A
        P = self.env.P
        R = self.env.r_stock
        gamma = self.env.gamma
        c = self.env.c
        self.cumulative_rewards = []

        V = np.zeros((len(S),))
        iteration = 0
        while True:
            delta = 0
            cumulative_gain = 0
            prev_V = V.copy()
            for sidx in range(len(S)):
                s = S[sidx]
                a = pi[sidx]
                p, sidx_next = self.env.get_transition_probabilities(a,s)
                r = self.env.get_transition_reward(a,s)
                V[sidx] = np.sum(p * (r + gamma * prev_V[sidx_next]))
                delta = np.max(np.abs(prev_V-V))

                cumulative_gain += V[sidx]
            self.cumulative_rewards.append(cumulative_gain)
            if  delta < epsilon:
                break
            iteration += 1
        return V
    
    def policy_improvement(self, V):
        S = self.env.S
        A = self.env.A
        P = self.env.P
        R = self.env.r_stock
        gamma = self.env.gamma
        c = self.env.c

        Q = np.zeros((len(S),len(A)), dtype=np.float64)

        for sidx in range(len(S)):
            s = S[sidx]
            for a in range(len(A)):
                p, sidx_next = self.env.get_transition_probabilities(a,s)
                r = self.env.get_transition_reward(a,s)
                Q[sidx][a] = np.sum(p*(r+gamma*V[sidx_next]))
        new_pi = {s: np.argmax(Q[s]) for s in range(len(S))}
        return new_pi
    
    def policy_iteration(self, epsilon):
        env = self.env
        A = env.A

        t = 0
        N = np.size(A)
        M = N*np.power(2,N)

        pi = {s: np.random.choice(A) for s in range(M)}
        while True:
            old_pi = pi.copy()
            V = self.policy_evaluation(pi,epsilon)
            pi = self.policy_improvement(V)
            t += 1
            if old_pi == pi:
                break
        return V, pi, t
    
    def plot_reward(self):
        plt.plot(self.cumulative_rewards)
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative Reward')
        plt.show()
    
class QLearningAgent:
    def __init__(self, env, alpha=0.1, epsilon=1, greedy_decay=0.995, episodes=1000):
        self.env = env
        self.alpha = alpha 
        self.gamma = env.gamma
        self.epsilon = epsilon
        self.greedy_decay = greedy_decay
        self.episodes = episodes
        self.Q = self.initialize_Q(env)
        self.accumulated_Q = []

    def initialize_Q(self, env):
        size_S = len(env.S)
        size_A = len(env.A)
        return np.zeros((size_S, size_A), dtype=np.float64)
    
    def eGreedy(self, s):
        if random.uniform(0,1) < self.epsilon:
            return random.choice(self.env.A)
        else:
            return np.argmax(self.Q[s])
        
    def learn(self, T=100):
        for episode in range(self.episodes):
            sid = self.env.reset()
            total_reward = 0
            #print(episode)
            t = 0
            while True:
                a = self.eGreedy(sid)
                s = self.env.S[sid]
                prob, sid_next = self.env.get_transition_probabilities(a,s)
                r = self.env.get_transition_reward(a,s)
                sid_prime = np.random.choice(sid_next, p = prob)
                s_prime = list(self.env.S.keys())[sid_prime]

                a_star = np.argmax(self.Q[sid_prime])
                gradient = r + self.gamma * self.Q[sid_prime][a_star] - self.Q[sid][a]
                self.Q[sid][a] += self.alpha * gradient

                sid = sid_prime

                total_reward += self.Q[sid][a]
                t += 1
                if(t == T*self.env.N):
                    break
            
            self.epsilon *= self.greedy_decay
            self.accumulated_Q.append(total_reward)

        return self.Q
    
    def get_policy(self):
        policy = {s: np.argmax(self.Q[s]) for s in range(len(self.env.S))}
        return policy
    
    def plot_reward(self):
        plt.plot(self.accumulated_Q, color='red', linewidth = 0.8)
        plt.xlabel("$Episode$")
        plt.ylabel("$Sum$ $of$ $rewards$")
        plt.show()


if __name__ == '__main__':

    ################ Environment ################
    N = 3
    env = MDPenvironment(N)
    print("State Space:", env.S);print("")
    print("Action Space:", env.A);print("")
    print("Markov Chain Probabilities:");print(env.P);print("")
    print("Markov Chain Rewards:");print(env.r_stock);print("")
    print("Discount Factor:", env.gamma);print("")
    print("Transaction fee:", env.c);print("")
    print("=======================================================================")
    
    ################ Model-Based Agent ################
    agent1 = Agent(env)
    V, pi1, t = agent1.policy_iteration(epsilon = 1e-10)

    print(f"Agent with full environment knowledge");print("")
    if N < 4:
        for i in range(N*2**N):
            print("For State", i,"| Optimal policy pi(s) =",pi1[i], "| Expected Reward Gt =", "{:.5f}".format(V[i]));print("")
    else:
        print(f"PI policy evaluated.")
    print("PI Algorithm Iterations: ", t)
    print("=======================================================================")

    agent1.plot_reward()


    ################ Model-Free Agent ################
    agent2 = QLearningAgent(env)
    Q = agent2.learn(T=150)
    pi2 = agent2.get_policy()

    print(f"Agent in a Model-free environment");print("")
    if N < 4:
        for i in range(N*2**N):
            print("For State", i,"| Optimal policy pi(s) =",pi2[i], "| Expected Reward Gt =", "{:.5f}".format(Q[i][pi2[i]]));print("")
    else: 
        print(f"Q-Learning policy evaluated.")
    print("=======================================================================")

    if(pi1 == pi2):
        print(f"Approximation succeeded.");print("")
    else:
        print(f"Approximation failed.");print("")

    agent2.plot_reward()
        
    