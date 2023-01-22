
import numpy as np
import random
from collections import defaultdict, deque


class QLearningAgent():

    def __init__(self, gamma, alpha):
        self.legalActions = []
        self.QValues = defaultdict(lambda: np.zeros((10, 10, 10, 10)))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (self.QValues[state][action]):
            return self.QValues[state][action]
        return 0.0

    def getLegalActions(self):
      if (len(self.legalActions)):
          return self.legalActions
      max_action = 50
      for i in range(1, max_action + 1):
        action = ()
        for i in range(0, 4):
            action += (random.randint(0, 9),)
        self.legalActions.append(action)
      return self.legalActions

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        qvalues = []
        for action in self.getLegalActions():
          qvalues.append(self.getQValue(state, action))
        if len(qvalues) == 0:
          return 0.0
        return max(qvalues)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        best_value = self.getValue(state)
        best_actions = [action for action in self.legalActions
                        if self.getQValue(state, action) == best_value]

        if len(best_actions) == 0:
          return None
        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        action = None
        action = random.choice(self.getLegalActions())
        # e-greedy for exploiting and exploring def
        # if util.flipCoin(self.epsilon):
        #     action = random.choice(legalActions)
        # else:
        #     action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward: float, alpha, gamma):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        qvalue = self.getQValue(state, action)
        next_value = self.getValue(nextState)

        self.QValues[(state, action)] = ((1-alpha) * qvalue) + \
            (alpha * (reward + gamma * next_value))

    def getPolicy(self, state):
        policy = self.computeActionFromQValues(state)
        self.legalActions = []
        return policy

    def getValue(self, state):
        return self.computeValueFromQValues(state)
