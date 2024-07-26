import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

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
            P = np.array([[0.65, 0.35, 0.1, 0.9],   #stock 0
                          [0.9, 0.1, 0.5, 0.5]])  #stock 1

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
            r = np.array([[0.08, -0.03],
                          [0.04,  0.01]])
        else:
            for i in range(N):
                r[i][0] = np.random.uniform(-0.02, 0.1)
                r[i][1] = np.random.uniform(-0.02, 0.1)

        return r.transpose()
    
    def step(self, a, s):
        N = self.N
        states = self.S
        P = self.P
        r_stock = self.r_stock
        c = self.c
        current_stock = s[0]
        current_state = s[a + 1]

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
        
        return p, args_next, R

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super(DQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear3 = nn.Linear(hidden_dim,action_dim)

    def forward(self, x):
        z1 = self.linear1(x)
        a1 = F.relu(z1)
        z2 = self.linear2(a1)
        a2 = F.relu(z2)
        y = self.linear3(a2)
        return y
    
class Agent:
    def __init__(self, env):
        self.env = env

    def policy_evaluation(self, pi, epsilon):
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
                p, sidx_next, r = self.env.step(a,s)
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
                p, sidx_next, r = self.env.step(a,s)
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
    def __init__(self, env, alpha=0.2, epsilon=1, episodes=1000):
        self.env = env
        self.alpha = alpha 
        self.gamma = env.gamma
        self.epsilon = epsilon
        self.greedy_decay = 0.0095
        self.min_epsilon = 0.05
        self.max_epsilon = 1
        self.episodes = episodes
        self.Q = self.initialize_Q(env)
        self.accumulated_Q = []
        self.accumulated_reward = []

    def initialize_Q(self, env):
        size_S = len(env.S)
        size_A = len(env.A)
        return np.zeros((size_S, size_A), dtype=np.float64)
    
    def eGreedy(self, s):
        if random.uniform(0,1) < self.epsilon:
            return random.choice(self.env.A)
        else:
            return np.argmax(self.Q[s])
        
    def learn(self):

        for episode in range(self.episodes):
            sid = self.env.reset()
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.greedy_decay*episode)
            t = 0
            T_eff = np.ceil(1/(1-self.gamma))
            # Parameters for plotting
            total_reward = 0
            total_Q = 0
            
            while True:
                s = self.env.S[sid]
                a = self.eGreedy(sid)
                prob, sid_next, r = self.env.step(a,s)
                sid_prime = np.random.choice(sid_next, p = prob)

                gradient = r + self.gamma * np.max(self.Q[sid_prime]) - self.Q[sid][a]
                self.Q[sid][a] += self.alpha * gradient
                
                sid = sid_prime

                total_reward += r
                total_Q += self.Q[sid][a]

                t += 1
                if(t == 10*self.env.N*T_eff):
                    break
            
            self.accumulated_Q.append(total_Q)
            self.accumulated_reward.append(total_reward)

        return self.Q
    
    def get_policy(self):
        policy = {s: np.argmax(self.Q[s]) for s in range(len(self.env.S))}
        return policy
    
    def plot_reward(self):
        plt.plot(self.accumulated_Q, color='red', linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Accumulated Q value")
        plt.show()
        plt.plot(self.accumulated_reward, color='blue', linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Sum of rewards")
        plt.show()

class DQNetworkAgent:
    def __init__(self, env, alpha=0.2, epsilon=1, episodes=1000):
        self.env = env
        self.alpha = alpha 
        self.gamma = env.gamma
        self.epsilon = epsilon
        self.greedy_decay = 0.0095
        self.min_epsilon = 0.05
        self.max_epsilon = 1
        self.episodes = episodes

        self.accumulated_Q = []
        self.accumulated_reward = []

        self.batch_size = 20
        self.target_freq = 50
        self.D = ReplayBuffer(size=10000)

        self.Q_Network =  DQNetwork(self.env.N + 1, len(self.env.A))
        self.T_network = DQNetwork(self.env.N + 1, len(self.env.A))
        self.T_network.load_state_dict(self.Q_Network.state_dict())
        self.T_network.eval()
        
        self.optimizer = torch.optim.SGD(self.Q_Network.parameters(), lr=self.alpha)
        self.cost = nn.MSELoss()


    def eGreedy(self, sid):
        if random.uniform(0,1) < self.epsilon:
            return random.choice(self.env.A)
        else:
            ##print(f"egreedy")
            with torch.no_grad():
                state = torch.FloatTensor(self.env.S[sid]).unsqueeze(0)
                q = self.Q_Network(state)
                return q.argmax().item()
            
    def learning_step(self):
        #if len(self.D) < self.batch_size:
        #    return

        sid = self.env.reset()
        for i in range(5):
            s = self.env.S[sid]
            a = self.eGreedy(sid)   #TODO: Change egreedy func in a way that decomposes a tensor and then choose an action likewise for get_policy()
            prob, sid_next, r = self.env.step(a,s)
            sid_prime = np.random.choice(sid_next, p = prob)
            s_prime = self.env.S[sid_prime]
            sid = sid_prime

            self.D.add((s,a,r,s_prime))

        print(self.D)
        
        B = self.D.sample(2)
        states, actions, rewards, next_states = zip(*B)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        Q_values = self.Q_Network(states)   #(batch_size, N-d arrays)
        print(Q_values)
        





    def learn(self):

        for episode in range(self.episodes):
            sid = self.env.reset()
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.greedy_decay*episode)
            T_eff = np.ceil(1/(1-self.gamma)) 
            t = 0
            total_reward = 0

            while True:
                s = self.env.S[sid]
                a = self.eGreedy(sid)   #TODO: Change egreedy func in a way that decomposes a tensor and then choose an action likewise for get_policy()
                prob, sid_next, r = self.env.step(a,s)
                sid_prime = np.random.choice(sid_next, p = prob)
                s_prime = self.env.S[sid_prime]

                self.D.add((s,a,r,s_prime))
                #print(self.D)

                sid = sid_prime

                if len(self.D) > self.batch_size:

                    B = self.D.sample(self.batch_size)
                    states, actions, rewards, next_states = zip(*B)

                    states = torch.FloatTensor(states)
                    actions = torch.LongTensor(actions)
                    rewards = torch.FloatTensor(rewards)
                    next_states = torch.FloatTensor(next_states)
                    '''
                    print(f"states shape: {states.shape}, dtype: {states.dtype}")
                    print(f"actions shape: {actions.shape}, dtype: {actions.dtype}")
                    print(f"rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
                    print(f"next_states shape: {next_states.shape}, dtype: {next_states.dtype}")
                    '''

                    Q_values = self.Q_Network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_Q_values = self.target_network(next_states).max(1)[0]
                    targets = rewards + self.gamma * next_Q_values


                    loss = self.cost(Q_values, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_reward += r

                if episode % self.target_freq == 0:
                    self.target_network.load_state_dict(self.Q_Network.state_dict())

                t += 1
                if(t == 3*self.env.N*T_eff):
                    break
            
            self.accumulated_reward.append(total_reward)
            print(f"Episode {episode} | Loss = {loss.item()} | Reward = {total_reward}")

    
    def get_policy(self):
        policy = {s: np.argmax(self.Q[s]) for s in range(len(self.env.S))}
        return policy
    
    def plot_reward(self):
        '''
        plt.plot(self.accumulated_Q, color='red', linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Accumulated Q value")
        plt.show()
        '''
        
        plt.plot(self.accumulated_reward, color='blue', linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Sum of rewards")
        plt.show()




if __name__ == '__main__':

    ################ Environment ################
    N = 3
    env = MDPenvironment(N)
    num = int(input("Press 1 if you want to see the Environment: "))
    if num == 1:
        print(f"State Space: {env.S} \n")
        print(f"Action Space: {env.A} \n")
        print(f"Markov Chain Probabilities: \n {env.P} \n")
        print(f"Markov Chain Rewards: \n {env.r_stock} \n")
        print(f"Discount Factor: {env.gamma} \n")
        print(f"Transaction fee: {env.c} \n")
    print("=======================================================================")
    
    ################ Model-Based Agent ################
    agent1 = Agent(env)
    V, pi1, t = agent1.policy_iteration(epsilon = 1e-10)

    print(f"Agent with full environment knowledge \n")
    if N < 1:
        for i in range(N*2**N):
            print(f"For State {i} | Optimal policy pi(s) = {pi1[i]} | Expected Reward Gt = {"{:.5f}".format(V[i])} \n")
    else:
        print(f"PI policy evaluated.")
    print(f"PI Algorithm Iterations: {t}")
    print("=======================================================================")


    ################ Model-Free Agent ################
    agent2 = QLearningAgent(env)
    Q = agent2.learn()
    pi2 = agent2.get_policy()

    print(f"Agent in a Model-free environment \n")
    if N < 1:
        for i in range(N*2**N):
            print(f"For State {i} | Optimal policy pi(s) = {pi2[i]} | Expected Reward Gt = {"{:.5f}".format(Q[i][pi2[i]])} \n")
    else: 
        print(f"Q-Learning policy evaluated.")
    
    print(Q)
    print("=======================================================================")

    agent2.plot_reward()

    ################ Neural Network Agent ################
    agent3 = DQNetworkAgent(env)
    #agent3.learn()
    #agent3.plot_reward()
    #agent3.learning_step()
        
    