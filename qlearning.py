
import numpy as np
from collections import defaultdict, deque
from itertools import chain

class ApproximateQLearning():

    def __init__(self):
        #TODO fix defaultdict size
        #TODO check maximum value if needed
        self.weights = defaultdict()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Q(state,action) = w * featureVector
        """
        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()
        dotProduct = features * weights
        return dotProduct

    def update(self, state, action, nextState, reward: float):
        """
          update your weights based on transition
        """
        #differece = reward + gamma*Q(s', a') - Q(s,a)
        difference = reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        weights = self.getWeights()
        #if weight vector is empty, initialize it to zero
        if len(weights) == 0:
            weights[state][action] = 0
        features = self.featExtractor.getFeatures(state, action)
        #iterate over features and multiply them by the learning rate (alpha) and the difference
        for key in features.keys():
            features[key] = features[key]*self.alpha*difference
        #sum the weights to their corresponding newly scaled features
        weights.__radd__(features)
        #update weights
        self.weights = weights.copy()

    def final(self, state):
        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            #print self.getWeights()
            pass
