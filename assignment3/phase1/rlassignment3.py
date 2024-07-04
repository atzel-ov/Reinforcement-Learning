import numpy as np
import matplotlib.pyplot as plt

class MDPenvironment:
    def __init__(self,N):
        self.N = N
        self.S = self.generete_state_space()
        self.A = [a for a in range(N)]
        self.P = self.generate_markov_probabilities()
        self.r_stock = self.generate_stock_rewards()
        self.gamma = 0.9
        self.c = 0.01

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
    
    def get_transition_probabilities(self,a,s):
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
    
    def get_transition_reward(self,a,s):
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
    def __init__(self,env):
        self.env = env

    def policy_evaluation(self,pi,epsilon,cumulative_rewards):
        S = self.env.S
        A = self.env.A
        P = self.env.P
        R = self.env.r_stock
        gamma = self.env.gamma
        c = self.env.c

        N = np.size(A)
        M = N * np.power(2, N)
        V = np.zeros((M,))
        iteration = 0
        while True:
            delta = 0
            cumulative_gain = 0
            prev_V = V.copy()
            for sidx in range(M):
                s = S[sidx]
                a = pi[sidx]
                p, sidx_next = self.env.get_transition_probabilities(a,s)
                r = self.env.get_transition_reward(a,s)
                V[sidx] = np.sum(p * (r + gamma * prev_V[sidx_next]))
                delta = np.max(np.abs(prev_V-V))

                cumulative_gain += V[sidx]
            cumulative_rewards.append(cumulative_gain)
            if  delta < epsilon:
                break
            iteration += 1
        return V
    
    def policy_improvement(self,V):
        S = self.env.S
        A = self.env.A
        P = self.env.P
        R = self.env.r_stock
        gamma = self.env.gamma
        c = self.env.c

        N = np.size(A)
        M = N*np.power(2,N)

        Q = np.zeros((M,N), dtype=np.float64)

        for sidx in range(M):
            s = S[sidx]
            for a in range(N):
                p, sidx_next = self.env.get_transition_probabilities(a,s)
                r = self.env.get_transition_reward(a,s)
                Q[sidx][a] = np.sum(p*(r+gamma*V[sidx_next]))
        new_pi = {s: np.argmax(Q[s]) for s in range(M)}
        return new_pi
    
    def policy_iteration(self,epsilon):
        env = self.env
        A = env.A

        t = 0
        N = np.size(A)
        M = N*np.power(2,N)
        cumulative_rewards = []

        pi = {s: np.random.choice(A) for s in range(M)}
        while True:
            old_pi = pi.copy()
            V = self.policy_evaluation(pi,epsilon,cumulative_rewards)
            pi = self.policy_improvement(V)
            t += 1
            if old_pi == pi:
                break
        return V, pi, t, cumulative_rewards
    

if __name__ == '__main__':

    N = 3
    env = MDPenvironment(N)
    agent = Agent(env)

    print("State Space:", env.S);print("")
    print("Action Space:", env.A);print("")
    print("Markov Chain Probabilities:");print(env.P);print("")
    print("Markov Chain Rewards:");print(env.r_stock);print("")
    print("Discount Factor:", env.gamma);print("")
    print("Transaction fee:", env.c);print("")

    V, pi, t, cumulative_rewards = agent.policy_iteration(epsilon = 1e-10)

    if N < 4:
        for i in range(N*2**N):
            print("For State", i,"| Optimal policy pi(s) =",pi[i], "| Expected Reward Gt =", "{:.5f}".format(V[i]));print("")

    print("PI Algorithm Iterations : ", t)
    plt.plot(cumulative_rewards)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Reward')
    plt.show()