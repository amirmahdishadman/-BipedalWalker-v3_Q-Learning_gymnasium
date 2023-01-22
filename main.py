import numpy as np
import random
import gymnasium as gym
import math
from collections import defaultdict, deque
import matplotlib.pyplot as graph
import os
ENV = "BipedalWalker-v3"
RENDER_MODE=str
EPISODES = 1000
GAMMA =  0.99
ALPHA=0.01
Load_Q_table=False


HIGHSCORE = -200

# stateBounds = [(0, math.pi),
#            (-2,2),
#            (-1,1),
#            (-1,1),
#            (0,math.pi),
#            (-2,2),
#            (0, math.pi),
#            (-2,2),
#            (0,1),
#            (0, math.pi),
#            (-2, 2),
#            (0, math.pi),
#            (-2, 2),
#            (0, 1)]


stateBounds=[
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



# actionBounds = (-1, 1)


def updateQTable (Qtable, state, action, reward, nextState=None):
    global GAMMA
    global ALPHA
    current = Qtable[state][action]  
    qNext = np.max(Qtable[nextState]) if nextState is not None else 0
    target = reward + (GAMMA * qNext)
    new_value = current + (ALPHA * (target - current))
    return new_value




#entekhab bar asas epsilon ke random bashe ya az ghabliya
def getNextAction(qTable, epsilon, state):

    if random.random() < epsilon:

        action = ()
        for i in range (0, 4):
            action += (random.randint(0, 9),)
    else:
        action = np.unravel_index(np.argmax(qTable[state]), qTable[state].shape)

    
    return action

def discretizeState(state):
    # print("*************************************************")
    # print(state)
    # print("#################################################")
    discreteState = []
    for i in range(len(state)):
        index = int((state[i]-stateBounds[i][0])  / (stateBounds[i][1]-stateBounds[i][0])*19)
        discreteState.append(index)
    return tuple(discreteState)


def convertNextAction(nextAction):
    action = []

    for i in range(len(nextAction)):

        nextVal = nextAction[i] / 9 * 2 - 1

        action.append(nextVal)
    # print(tuple(action))
    return tuple(action)

def plotEpisode(myGraph, xval, yval, epScore, plotLine, i):

    xval.append(i)
    yval.append(epScore)

    plotLine.set_xdata(xval)
    plotLine.set_ydata(yval)
    myGraph.savefig("./plot")


#first function to run
def runAlgorithmStep(env, i, qTable):

    global HIGHSCORE

    env_R=env.reset()[0:24]
    env_R=env_R[0]
    state = discretizeState(env_R)
    total_reward=  0


    #setting epsilon
    if(Load_Q_table):
        epsilon = 0.1
    else:
        epsilon = 1.0 / ( i * .004)
    print("Epsilon: "+ str(epsilon))
    
    iterate=0
    while True:
        iterate+=1
        nextAction = convertNextAction(getNextAction(qTable, epsilon, state))
        nextActionDiscretized = getNextAction(qTable, epsilon, state)
        nextState, reward, done, nn, info = env.step(nextAction)
        nextState = discretizeState(nextState[0:14])
        total_reward += reward
        qTable[state][nextActionDiscretized] = updateQTable(qTable, state, nextActionDiscretized, reward, nextState)
        state = nextState
        if done:
            break
        if(iterate>500):
            break
    print("total_revard= "+str(total_reward))
    if total_reward > HIGHSCORE:

        HIGHSCORE = total_reward

    return total_reward
    
def main():
    
    global HIGHSCORE
    
    load=input("if you want load enter file name else enter no:\n")
    save_path=input("enter the name you want save file as : \n")
    if(load=="no"):
        Load_Q_table=False
    else:
        loaded_qtable = np.load(load,allow_pickle=True)
        Load_Q_table=True

    
    visualize = input("Visualize? [y/n]\n")

    if visualize == 'y':
        RENDER_MODE = "human"
    else:
        RENDER_MODE = None

    env = gym.make(ENV,render_mode=RENDER_MODE)

    qTable = defaultdict( lambda: np.zeros((10, 10, 10, 10)))
    if(Load_Q_table):
        qTable.update(loaded_qtable.item())

    myGraph = graph.figure()
    xval, yval = [], []
    mySubPlot = myGraph.add_subplot()
    graph.xlabel("Episode #")
    graph.ylabel("Score")
    graph.title("Scores vs Episode")
    plotLine, = mySubPlot.plot(xval, yval)
    mySubPlot.set_xlim([0, EPISODES])
    mySubPlot.set_ylim([-220, 10])

    
    for i in range(1, EPISODES + 1):

        epScore = runAlgorithmStep(env, i, qTable)
        print("Episode "+str(i)+" finished. Now plotting..\n")
        plotEpisode(myGraph, xval, yval, epScore, plotLine, i)
        print("--------------------------------------------------------\n")

        #save in every 20 episode
        # if(i%100==0):
            # np.save(save_path, np.array(dict(qTable)))
        
    
    print("All episodes finished. Highest score achieved: " + str(HIGHSCORE))
    np.save(save_path, np.array(dict(qTable)))
    # print(qTable)     
    env
    

  
main()