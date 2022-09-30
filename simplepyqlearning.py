import numpy as np
import random as rd
from tqdm import tqdm


def get_value_from_array(array, a_index, equal=None):
    a = array
    for i in reversed(range(len(a_index))):
        if equal == True:
            print(a)
        a = a[a_index[i]]
    return a


class QLearning:
    def __init__(self, action, map, terminal_state, epsylone=0.7, epsylone_dec=0.01, epsylone_min=0.1,
                 learning_rate=0.8, discount_factor=0.5, spawn=True):
        self.epsylone_min = epsylone_min
        self.epsylone_dec = epsylone_dec
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.terminal_state = terminal_state
        self.spawn = spawn
        self.map = np.array(map)
        if self.spawn == True:
            self.spawn = np.random.randint(0, self.map.shape[0], self.map.shape)
        else:
            self.spawn = np.array(self.spawn)

        self.epsylone = epsylone
        self.action = np.array(action)
        self.q_value = np.zeros(np.concatenate((self.map.shape, np.array([len(action)]))))

    def action_is_possible(self, s, a):
        new_s = s - self.action[a]
        try:
            test = self.map[tuple(new_s)]
            return True
        except:
            return False

    def get_next_state(self, s, a):
        return s - a

    def is_terminal_state(self, s):
        for i in self.terminal_state:
            if self.map[tuple(s)] == i:
                return True
        return False

    def get_all_cheminement(self, spawn):
        s = spawn
        action = []
        state = []
        while not self.is_terminal_state(s):
            a = np.array(rd.randint(0, self.action.shape[0] - 1))
            action.append(a)
            s = self.get_next_state(s, self.action[a])
            state.append(s)

        print(self.map[tuple(s)])
        return action, state, self.map[tuple(s)]

    def train(self, episode):

        for i in tqdm(range(episode)):
            s = self.spawn
            while not self.is_terminal_state(s):
                while True:
                    if rd.random() > self.epsylone:

                        a = np.argmax(self.q_value[tuple(s)])

                    else:

                        a = np.array(rd.randint(0, self.action.shape[0] - 1))
                    if self.action_is_possible(s, a):
                        break

                old_s = s
                s = self.get_next_state(s, self.action[a])

                r = self.map[tuple(s)]
                old_q_value = self.q_value[tuple(old_s)][a]
                temporal_difference = r + (
                        self.discount_factor * np.max(self.q_value[tuple(old_s)])) - old_q_value
                new_q_value = old_q_value + (self.learning_rate * temporal_difference)
                # print(old_s, self.learning_rate, temporal_difference, new_q_value)
                self.q_value[tuple(old_s)][a] = new_q_value

            if self.epsylone > self.epsylone_min:
                self.epsylone -= self.epsylone_dec

        print(self.q_value, self.epsylone)
        return self.q_value


map = [[[-1, 0, 0, 0, 1],
       [1, 0, 0, 0, -1]]]
q = QLearning(action=[[0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], map=map, terminal_state=[-1, 1], epsylone=1,
              epsylone_dec=0.001, epsylone_min=0.01,
              learning_rate=0.8, discount_factor=1, spawn=[0, 0, 2])
q.train(1000)
print(q.get_all_cheminement([0, 2]))
