
import numpy as np
import random
from collections import defaultdict, deque
from itertools import chain

class QLearningAgent():
    epsilon=1
    def __init__(self, gamma, alpha,loaded_qtable,Load_Q_table):
        self.legalActions = []
        self.QTable = defaultdict(lambda: np.zeros((10, 10)))
        if(Load_Q_table):
          self.QTable.update(loaded_qtable.item())
          self.epsilon=0.3

    def getQValue(self, state):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        
        if (self.QTable[state].any()):
            return self.QTable[state]
        return 0.0

#behine sazi action
    def getLegalActions(self,state):
      # if (len(self.legalActions)):
      #     return self.legalActions
      # max_action = 50
      # for i in range(1, max_action + 1):
      #   action = ()
      #   for i in range(0, 2):
      #       action += (random.randint(0, 9),)
      #   self.legalActions.append(action)
      # return self.legalActions

      if random.random() < self.epsilon:
        action = ()
        for i in range (0, 2):
            action += (random.randint(0, 9),)
      else:
        action = np.unravel_index(np.argmax(self.QTable[state]), self.QTable[state].shape)

      return action



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # action = self.getLegalActions(state)
        qvalues=(self.getQValue(state))
        
        if (qvalues) == 0.0:
          return 0.0
        print(qvalues)
        return max(chain.from_iterable(qvalues))

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

        # Pick Action
        action = None
        action = self.getLegalActions(state)
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
        
        qvalue = self.QTable[state][action]
        next_value = self.getValue(nextState)
        self.QTable[state][action] = ((1-alpha) * qvalue) + \
            (alpha * (reward + gamma * next_value))

        # print(self.QValues[state][action])
        if(self.epsilon>0):
          self.epsilon-=0.001




    def getPolicy(self, state):
        policy = self.computeActionFromQValues(state)
        self.legalActions = []
        return policy

    def getValue(self, state):
        return self.computeValueFromQValues(state)
