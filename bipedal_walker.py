class BipedalWalker():
    def __init__(self):
        self.highscore = -1000
        self.stateBounds = [
                            (-3.14, 3.14),
                            (-5.0, 5.0),
                            (-5.0, 5.0),
                            (-5.0, 5.0),
                            (-3.14, 3.14),
                            (-5.0, 5.0),
                            (-3.14, 3.14),
                            (-5.0, 5.0),
                            (-0.0, 5.0),
                            (-3.14, 3.14),
                            (-5.0, 5.0),
                            (-3.14, 3.14),
                            (-5.0, 5.0),
                            (-0.0, 5.0),
                            (-1.0, 1.0),
                            (-1.0, 1.0),
                            (-1.0, 1.0),
                            (-1.0, 1.0),
                            (-1.0, 1.0),
                            (-1.0, 1.0),
                            (-1.0, 1.0),
                            (-1.0, 1.0),
                            (-1.0, 1.0),
                            (-1.0, 1.0)
                            ]

    def normalizeState(self, state):
        discreteState = []
        for i in range(len(state)):
            index = int((state[i]-self.stateBounds[i][0]) / (self.stateBounds[i][1]-self.stateBounds[i][0])*19)
            discreteState.append(index)
        return tuple(discreteState)

    def denormalizeAction(self, nextAction):
      action = []
      for i in range(len(nextAction)):
          nextVal = nextAction[i] / 9 * 2 - 1
          action.append(nextVal)
      return tuple(action)
