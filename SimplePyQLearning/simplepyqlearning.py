import random
import random as rd

import numpy as np
from tqdm import tqdm

color1 = '\033[92m'
color2 = '\033[94m'
color3 = '\033[0m'

print(
    f"{color1}You use the library SimplePyQLearning !\n{color2}Credit: LeLaboDuGame on Twitch -> "
    f"https://twitch.tv/LeLaboDuGame{color3}")



def something(self, reward, state, action):
    """
        :return: reward, action, state
        :param self: return all variables
        :param reward: return the reward
        :param action: return the action
        :param state: return the state


    Exemple:
        print(self.reward)

        self.reward = 1
        return reward, action, state
    """
    return reward, state, action


class QLearning:
    """
    SIMPLEPYQLEARNING
    -----------------
    Hey guy !
    I'm 15 years frensh dev !
    I present you my project of q-qlearning !
    Just for the credit : LeLaboDuGame on Twitch -> https://twitch.tv/LeLaboDuGame
    You can use this library on all your project !

    Exemple:
    -------
        map = [[0, -1, 0,  0, 1],
       [0, -1, 0, -1, 0],
       [0, -1, 0, -1, 0],
       [0, -1, 0, -1, 0],
       [0,  0, 0, -1, 0]]
        q = QLearning(action=[[-1, 0], [1, 0], [0, -1], [0, 1]], map=map, terminal_state=[-1, 1], epsylone=1,
              epsylone_dec=0.0001, epsylone_min=0.1,
              learning_rate=0.8, discount_factor=0.8, spawn="rnd")
        q.train(10000)
        print(q.get_all_cheminement([0, 0], timeout=1000, debug=False))
    """

    def __init__(self, action, map, terminal_state, epsylone=0.7, epsylone_dec=0.001, epsylone_min=0.1,
                 learning_rate=0.8, discount_factor=0.8, spawn="rnd"):
        self.map = np.array(map)
        self.epsylone_min = epsylone_min
        self.epsylone_dec = epsylone_dec
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.terminal_state = terminal_state
        self.spawn = spawn
        if self.spawn == "random" or self.spawn == "rnd":
            self.spawn = self.random_spawn()

        print(f"Initiale spawn is {self.spawn}.")

        self.spawn = (np.array(self.spawn), spawn == "rnd")

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

    def random_spawn(self):
        while True:
            spawn = []
            for i in self.map.shape:
                spawn.append(random.randint(0, i - 1))
            if not self.is_terminal_state(spawn):
                break
        return spawn

    def is_terminal_state(self, s):
        for i in self.terminal_state:
            try:
                if self.map[tuple(s)] == i:
                    return True
            except:
                return False
        return False

    def get_all_cheminement(self, spawn, timeout=10000, debug=True):
        s = spawn
        actions = []
        states = [s]
        i = 0
        while not self.is_terminal_state(s):
            if i == timeout:
                print(
                    f"{color1}Stoped because to much repetition.\nMaybe it's because it has a probleme somewhere. Verify the terminal_state values.{color3}")
                timeout = True
                break
            x = 1
            while True:
                a = np.argsort(self.q_value[tuple(s)])[-x]
                if self.action_is_possible(s, a):
                    break
                x += 1
            actions.append((a.tolist(), self.action[a].tolist()))
            s = self.get_next_state(s, a)
            states.append(s.tolist())
            i += 1
        if timeout == True:
            if not debug:
                return None
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

    def train(self, episode, custom_function=something):

        for i in tqdm(range(episode)):
            if self.spawn[1]:
                s = self.random_spawn()
            else:
                s = self.spawn[0]
            while not self.is_terminal_state(s):
                rnd = rd.random()
                x = 1
                while True:
                    if np.zeros((len(self.action))).tolist() == self.q_value[tuple(s)].tolist():
                        a = np.array(rd.randint(0, self.action.shape[0] - 1))
                    else:
                        if rnd > self.epsylone:
                            a = np.argsort(self.q_value[tuple(s)])[-x]
                        else:
                            a = np.array(rd.randint(0, self.action.shape[0] - 1))
                    if self.action_is_possible(s, a):
                        break
                    x += 1

                old_s = s
                s = self.get_next_state(s, a)
                r = self.map[tuple(s)]
                r, s, a = custom_function(self, r, s, a)
                old_q_value = self.q_value[tuple(old_s)][a]
                temporal_difference = r + (
                        self.discount_factor * np.max(self.q_value[tuple(s)])) - old_q_value
                new_q_value = old_q_value + self.learning_rate * temporal_difference
                # print(old_s, self.learning_rate, temporal_difference, new_q_value)
                self.q_value[tuple(old_s)][a] = new_q_value

            if self.epsylone > self.epsylone_min and episode % 10 == 0:
                self.epsylone -= self.epsylone_dec
        print(self.q_value, self.epsylone)
        return self.q_value
