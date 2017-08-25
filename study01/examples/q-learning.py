import numpy as np
import random
import sys

R = np.array([
[-1, -1, -1, -1,  0,  -1],
[-1, -1, -1,  0, -1, 100],
[-1, -1, -1,  0, -1,  -1],
[-1,  0,  0, -1,  0,  -1],
[ 0, -1, -1,  0, -1, 100],
[-1,  0, -1, -1,  0, 100]
])

Q = np.zeros((6,6))
LEARNING_COUNT = 1000
GAMMA = 0.8
GOAL_STATE = 5

class QLearning(object):
    def __init__(self):
        return

    def learn(self):
        state = self._getRandomState()
        for i in range(LEARNING_COUNT):
            possible_actions = self._getPossibleActionsFromState(state)

            action = random.choice(possible_actions)

            next_state = action
            next_possible_actions = self._getPossibleActionsFromState(next_state)
            max_Q_next_s_a = self._getMaxQvalueFromStateAndPossibleActions(next_state, next_possible_actions)
            Q[state, action] = R[state, action] + GAMMA * max_Q_next_s_a

            state = next_state

            if state == GOAL_STATE:
                state = self._getRandomState()

    def _getRandomState(self):
        return random.randint(0, R.shape[0] - 1)

    def _getPossibleActionsFromState(self, state):
        if state < 0 or state >= R.shape[0]: sys.exit("部屋: %d へは行けません" % state)
        return list(np.where(np.array(R[state] != -1)))[0]

    def _getMaxQvalueFromStateAndPossibleActions(self, state, possible_actions):
        return max([Q[state][i] for i in (possible_actions)])

    def dumpQvalue(self):
        print (Q.astype(int))

    def runGreedy(self, start_state = 0):
        print ("===== スタート =====")
        state = start_state
        while state != GOAL_STATE:
            print ("現在の部屋: %d" % state)
            possible_actions = self._getPossibleActionsFromState(state)

            max_Q = 0
            best_action_candidates = []
            for a in possible_actions:
                if Q[state][a] > max_Q:
                    best_action_candidates = [a,]
                    max_Q = Q[state][a]
                elif Q[state][a] == max_Q:
                    best_action_candidates.append(a)

            best_action = random.choice(best_action_candidates)
            print ("-> 行動: %d に向かう" % best_action)
            state = best_action
        print ("部屋: %d, ゴール!!" % state)

if __name__ == "__main__":
    QL = QLearning()
    QL.learn()

    QL.dumpQvalue()

    for s in range(R.shape[0]-1):
        QL.runGreedy(s)