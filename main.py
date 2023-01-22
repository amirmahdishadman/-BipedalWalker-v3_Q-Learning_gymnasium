import matplotlib.pyplot as graph
import gymnasium as gym
from bipedal_walker import BipedalWalker
from qlearning import QLearningAgent
import numpy as np

#add for load saved q-tables:
Load_Q_table=False




MAX_EPISODES = 1000
MAX_TIMESTEPS = 500
gamma = 0.99
alpha = 0.1
bipedalWalker = BipedalWalker()


#saving commands
# load=input("if you want load enter file name else enter no:\n")
# save_path=input("enter the name you want save file as : \n")


# if(load=="no"):
#     Load_Q_table=False
# else:
#     loaded_qtable = np.load(load,allow_pickle=True)
#     Load_Q_table=True



visualize = input("Visualize? [y/n]\n")
if visualize == 'y':
    RENDER_MODE = "human"
else:
    RENDER_MODE = None
env = gym.make("BipedalWalker-v3", render_mode=RENDER_MODE)
state , info = env.reset(seed=42)
qLearning = QLearningAgent(gamma, alpha)

myGraph = graph.figure()
xval, yval = [], []
mySubPlot = myGraph.add_subplot()
graph.xlabel("Episode #")
graph.ylabel("Score")
graph.title("Scores vs Episode")
plotLine, = mySubPlot.plot(xval, yval)
mySubPlot.set_xlim([0, MAX_EPISODES])
mySubPlot.set_ylim([-200, 300])

def plotEpisode(myGraph, xval, yval, epScore, plotLine, i):

    xval.append(i)
    yval.append(epScore)
    plotLine.set_xdata(xval)
    plotLine.set_ydata(yval)
    myGraph.savefig("./ScoreEpisodePlot")

def runEpisode(MAX_TIMESTEPS, bipedalWalker, env, qLearning, episode):
    state = bipedalWalker.normalizeState(env.reset()[0:14][0])
    total_reward = 0
    print(f"\n\nEpisode {episode}: ========================================================================")
    for step in range(MAX_TIMESTEPS):
        action = qLearning.getAction(state)
        next_state , reward , terminated , truncated , _ = env.step(bipedalWalker.denormalizeAction(action))
        next_state = bipedalWalker.normalizeState(next_state[0:14])
        qLearning.update(state, action, next_state, reward, alpha, gamma)
        print(f"\t\treward: {reward} for action {action}")
        total_reward += reward
        done = terminated or truncated or step == MAX_TIMESTEPS -1
        if done:
            # TODO [We can update the policy here if we use Policy extraction]
            state , info = env.reset()
            break
    print(f"total_reward = {total_reward}")
    if total_reward > bipedalWalker.highscore:
        bipedalWalker.highscore = total_reward
    return total_reward



# ========================================= RUN ALGORITHM =========================================

for episode in range(1, MAX_EPISODES + 1):
    if (episode > (MAX_EPISODES / 3) - 200): 
        alpha = 0.5
    if (episode > (MAX_EPISODES * 2 / 3)): alpha = 0.9
    episode_score = runEpisode(MAX_TIMESTEPS, bipedalWalker, env, qLearning, episode)
    plotEpisode(myGraph, xval, yval, episode_score, plotLine, episode)
    print("plot_ends \n ________________________________________________________________")


#saving qvalues
# np.save(save_path, np.array(dict(qLearning.QValues)))
env.close()