import random as rd

import numpy as np
from tqdm import tqdm


class QLearning:
    """

    SimplePyQLearning:
    ----------------
    Create by LeLaboDuGame

    Exemple:
    -------
        map = [[0, -1, 0, 0, 1],
        [0, 0, 0, -1, 0]]
        q = QLearning(action=[[-1, 0], [1, 0], [0, -1], [0, 1]], map=map, terminal_state=[-1, 1], epsylone=1,
        epsylone_dec=0.0001, epsylone_min=0.1,
        learning_rate=0.8, discount_factor=1, spawn=[0, 0])
        q.train(1000)
        print(q.get_all_cheminement([0, 0]))
    """

    def __init__(self, action, map, terminal_state, epsylone=0.7, epsylone_dec=0.01, epsylone_min=0.1,
                 learning_rate=0.8, discount_factor=0.5, spawn=[]):
        self.map = np.array(map)
        self.epsylone_min = epsylone_min
        self.epsylone_dec = epsylone_dec
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.terminal_state = terminal_state
        self.spawn = spawn
        if self.spawn == []:
            for i in self.map.shape:
                self.spawn.append(i)

        self.spawn = (np.array(self.spawn), spawn == [])

        self.epsylone = epsylone
        self.action = np.array(action)
        self.q_value = np.zeros(np.concatenate((self.map.shape, np.array([len(action)]))))

    def action_is_possible(self, s, a):
        new_s = s - self.action[a]
        for i in range(len(new_s)):
            if new_s[i] < 0 or new_s[i] >= self.map.shape[i]:
                return False
        return True

    def get_next_state(self, s, a):
        return s - self.action[a]

    def is_terminal_state(self, s):
        for i in self.terminal_state:
            try:
                if self.map[tuple(s)] == i:
                    return True
            except:
                return False
        return False

    def get_all_cheminement(self, spawn):
        s = spawn
        actions = []
        states = [s]
        while not self.is_terminal_state(s):
            if np.argmax(self.q_value[tuple(s)]) == 0:
                while True:
                    a = np.array(rd.randint(0, self.action.shape[0] - 1))
                    if self.action_is_possible(s, a) and not self.is_terminal_state(self.get_next_state(s, a)):
                        break
            else:
                a = np.argmax(self.q_value[tuple(s)])
            actions.append((a.tolist(), self.action[a].tolist()))
            s = self.get_next_state(s, a)
            states.append(s.tolist())

        print(self.show_player_pass(states))
        return actions, states, self.map[tuple(s)]

    def show_player_pass(self, states):
        """

        :param states:
        :return: passage of the player: S = Spawn; P = Passage; E = End
        """
        map = np.array(self.map.tolist(), dtype=str)
        for s in states:
            if states.index(s) == 0:
                map[tuple(s)] = "S"
            elif states.index(s) == len(states) - 1:
                map[tuple(s)] = "E"
            else:
                map[tuple(s)] = "P"
        return map

    def train(self, episode):

        for i in tqdm(range(episode)):
            s = self.spawn[0]
            if self.spawn[1]:
                print(f"Repetition n°{i} at spawn : {s}")
            while not self.is_terminal_state(s):
                while True:
                    if np.zeros((len(self.action))).tolist() == self.q_value[tuple(s)].tolist():
                        a = np.array(rd.randint(0, self.action.shape[0] - 1))
                    else:
                        if rd.random() > self.epsylone:
                            a = np.argmax(self.q_value[tuple(s)])
                        else:
                            a = np.array(rd.randint(0, self.action.shape[0] - 1))
                    if self.action_is_possible(s, a):
                        break
                old_s = s
                s = self.get_next_state(s, a)
                r = self.map[tuple(s)]
                old_q_value = self.q_value[tuple(old_s)][a]
                temporal_difference = r + (
                        self.discount_factor * np.max(self.q_value[tuple(s)])) - old_q_value
                new_q_value = old_q_value + self.learning_rate * temporal_difference
                # print(old_s, self.learning_rate, temporal_difference, new_q_value)
                self.q_value[tuple(old_s)][a] = new_q_value

            if self.epsylone > self.epsylone_min and episode % 10:
                self.epsylone -= self.epsylone_dec
        print(self.q_value, self.epsylone)
        return self.q_value
