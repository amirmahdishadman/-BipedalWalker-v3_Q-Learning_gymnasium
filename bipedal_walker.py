class BipedalWalker():
    binery=0
    binery2=0
    def __init__(self):
        self.highscore = -1000
        self.stateBounds =[
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
            # index = int((state[i]-self.stateBounds[i][0]) / (self.stateBounds[i][1]-self.stateBounds[i][0])*19)
            index = int((state[i]-self.stateBounds[i][0]) / (self.stateBounds[i][1]-self.stateBounds[i][0])*19)
            # print(index)
            discreteState.append(index)
        return tuple(discreteState)

    def denormalizeAction(self, nextAction):
      
      action = []
      actions_final=[]
      for i in range(len(nextAction)):
          nextVal = nextAction[i] / 9 * 2 - 1
          action.append(nextVal)


      actions_final.append(0.5)  
      actions_final.append(action[0])
      actions_final.append(-0.5)
      actions_final.append(action[1])
      self.binery+=1
      if(self.binery>50):
        actions_final[0]=-0.5
        actions_final[2]=0.5
        self.binery2+=1
      if(self.binery2==50):
        self.binery=0
        self.binery2=0
    #   print(actions_final)
      return tuple(actions_final)
