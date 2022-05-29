import numpy as np
import pandas as pd
import time
import read_maze
class Qlearning():
    def __init__(self, N_STATES=6, EPSILON: float = 0.9, ALPHA: float = 0.2,
                 GAMMA: float = 0.9, MAX_EPISODES: int = 13,
                 FRESH_TIME: float = 0.3):
        np.random.seed(2)                     # reproducible
        self.N_STATES = N_STATES              # the dimension of maze
        self.ACTIONS = ['W', 'A', 'S', 'D']   # action of agent
        # self.ACTIONS = ['left', 'right']
        self.EPSILON = EPSILON                # greedy
        self.ALPHA = 0.3                      # learning rate
        self.GAMMA = GAMMA                    # gamma
        self.MAX_EPISODES = MAX_EPISODES      # max episoes
        self.FRESH_TIME = 0.3                 # interal time
        self.position = [1, 1]
        self.flag = 0

    def build_q_table(self):                  #build the Q talbe
        n_states = self.N_STATES
        actions = self.ACTIONS
        table = pd.DataFrame(columns=['State']+actions)
        self.q_table = table

    def set_State(self, x:int, y: int):
        S = []
        # info = read_maze.get_local_maze_information(x, y)
        # w_info = info[0][1]
        # a_info = info[1][0]
        # s_info = info[2][1]
        # d_info = info[1][2]
        S.append(x)
        S.append(y)
        # if(w_info[1] == 0):
        #     S.append(w_info[1])
        # else:
        #     S.append(1)
        # if(a_info[1] == 0):
        #     S.append(a_info[1])
        # else:
        #     S.append(1)
        # if(s_info[1] == 0):
        #     S.append(s_info[1])
        # else:
        #     S.append(1)
        # if(s_info[1] == 0):
        #     S.append(s_info[1])
        # else:
        #     S.append(1)
        return S

    def choose_action(self, state):     # choose action
        random = (self.flag > 8)
        state_actions = (self.q_table.loc[self.q_table['State'] == str(state)]).iloc[:, 1:]  # select the all q values
        if (np.random.uniform() > self.EPSILON) or (state_actions.all() == 0).all() or random:  #ungreedy or null state
            action_name = np.random.choice(self.ACTIONS)
        else:
            state_actions = self.q_table.loc[self.q_table['State'] == str(state), :].iloc[0, 1:].astype('float64') 
            # select the all q values
            action_name = state_actions.idxmax()    # greedy
        if random:
            self.flag = 0
        return action_name

    def get_env_feedback(self, S, A):    # get feedback: S' and Reward R
        R = 0
        info = read_maze.get_local_maze_information(self.position[0], self.position[1])
        w_info = info[0][1]
        a_info = info[1][0]
        s_info = info[2][1]
        d_info = info[1][2]
        if A == 'W':
            if(w_info[0] == 1 and w_info[1] == 0):
                self.position[0] -= 1
            if(w_info[0] == 0):
                R = -1
            if(w_info[1] != 0):
                R = -0.7
        if A == 'A':
            if(a_info[0] == 1 and a_info[1] == 0):
                self.position[1] -= 1
            if(a_info[0] == 0):
                R = -1
            if(a_info[1] != 0):
                R = -0.7
        if A == 'S':
            if(s_info[0] == 1 and s_info[1] == 0):
                self.position[0] += 1
            if(s_info[0] == 0):
                R = -1
            if(s_info[1] != 0):
                R = -0.7
        if A == 'D':
            if(d_info[0] == 1 and w_info[1] == 0):
                self.position[1] += 1
            if(d_info[0] == 0):
                R = -1
            if(d_info[1] != 0):
                R = -0.7
        if(self.position[0] == 200 and self.position[1] == 200):
            S_ = 'terminal'
            R = 1
        else:
            S_ = self.set_State(self.position[0], self.position[1])
        return S_, R

    def update_env(self, S, episode, step_counter):
        if S == 'terminal':
            interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
            print('{}'.format(interaction))
            # time.sleep(2)

    def learn(self):
        self.build_q_table() #build q_table
        for episode in range(self.MAX_EPISODES): 
            read_maze.load_maze()
            step_counter = 0
            self.position = [1, 1]
            S = self.set_State(1, 1)
            is_terminated = False
            self.update_env(S, episode, step_counter) 
            while not is_terminated:
                self.EPSILON = self.EPSILON * 0.99
                if(not str(S) in self.q_table['State'].values):
                    self.q_table = self.q_table.append({'State': str(S), 'W': 0., 'A': 0.,
                                         'S': 0., 'D': 0.}, ignore_index=True)
                q_table_before = self.q_table.shape[0]
                A = self.choose_action(S) #Choose action 
                S_, R = self.get_env_feedback(S, A) # Do the action and get the feed back.
                q_predict = float(self.q_table.loc[self.q_table['State'] == str(S), A]) # predict the Q value
                if(not str(S_) in self.q_table['State'].values):
                    self.q_table = self.q_table.append({'State': str(S_), 'W': 0., 'A': 0.,
                                         'S': 0., 'D': 0.}, ignore_index=True)
                q_table_after = self.q_table.shape[0]
                if(q_table_after-q_table_before == 0):
                    self.flag += 1
                if S_ != 'terminal':
                    q_target = R + self.GAMMA * float((self.q_table.loc[self.q_table['State'] == str(S_)]).iloc[:, 1:].max(axis=1)) #update the Q value
                else:
                    q_target = R # Actual Q value
                    is_terminated = True
                if (step_counter//100 == 0):
                    self.ALPHA = self.ALPHA * 0.9
                self.q_table.loc[self.q_table['State'] == str(S), A] += self.ALPHA*(q_target - q_predict) #q_table update 
                S = S_ 

                self.update_env(S, episode, step_counter+1)

                step_counter += 1
                self.q_table.to_csv("Qtable_data_4_xy.csv")
                data=open("data.txt",'a+')
                print("Action: ", A,file=data)
                print('position(x, y)', self.position,file=data)
                info = read_maze.get_local_maze_information(self.position[0], self.position[1])
                print('Wall and Fire',file=data)
                print(info[0][0][0], info[0][1][0], info[0][2][0],' | ', info[0][0][1], info[0][1][1], info[0][2][1],file=data)
                print(info[1][0][0], info[1][1][0], info[1][2][0],' | ', info[1][0][1], info[1][1][1], info[1][2][1],file=data)
                print(info[2][0][0], info[2][1][0], info[2][2][0],' | ', info[2][0][1], info[2][1][1], info[2][2][1],file=data)
                print('----------------------------------------', file=data)
                data.close()
        return self.q_table


if __name__ == "__main__":
    agent = Qlearning()
    agent.learn()