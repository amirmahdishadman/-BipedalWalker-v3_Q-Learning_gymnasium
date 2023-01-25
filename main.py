import matplotlib.pyplot as graph
import gymnasium as gym
from bipedal_walker import BipedalWalker
from qlearning import QLearningAgent
import numpy as np

#add for load saved q-tables:
Load_Q_table=False




MAX_EPISODES = 1000
MAX_TIMESTEPS = 1500
gamma = 0.9
alpha = 0.1
bipedalWalker = BipedalWalker()


#saving commands
load=input("if you want load enter file name else enter no:\n")
test=input(" do you want to test? (y,n) ")
save_path=input("enter the name you want save file as : \n")



if(load=="no"):
    Load_Q_table=False
    loaded_qtable=[]
else:
    loaded_qtable = np.load(load,allow_pickle=True)
    Load_Q_table=True


record_video=False
record=input("record 1500 Episode? (yes) (no)\n")
visualize = input("Visualize? [y/n]\n")
if visualize == 'y':
    render_mode = "human"
else:
    render_mode = None


if(record=="yes"):
    render_mode="rgb_array"
env = gym.make("BipedalWalker-v3", render_mode=render_mode)
if(record=="yes"):
    env = gym.wrappers.RecordVideo(env, 'video', step_trigger = lambda x: x<=1500, name_prefix='output', video_length=1500)
qLearning = QLearningAgent(gamma, alpha,loaded_qtable,Load_Q_table)


if(test=="y"):
    alpha=0.0
    qLearning.epsilon=0.0





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
    state = bipedalWalker.normalizeState(env.reset()[0][0:24])
    total_reward = 0
    max_reward=-100
    print(f"\n\nEpisode {episode}: ========================================================================")
    for step in range(MAX_TIMESTEPS):
        action = qLearning.getAction(state)
        # print(action)
        next_state , reward , terminated , truncated , _ = env.step(bipedalWalker.denormalizeAction(action))
        next_state = bipedalWalker.normalizeState(next_state[0:24])
        qLearning.update(state, action, next_state, reward, alpha, gamma)
        if(reward>max_reward):
            max_reward=reward
        # print(f"\t\treward: {reward} for action {action}")
        total_reward += reward
        done = terminated or truncated or step == MAX_TIMESTEPS -1
        if done:
            # TODO [We can update the policy here if we use Policy extraction]
            state , info = env.reset()
            break
    print(f"total_reward = {total_reward}")
    print("max_reward : "+str(max_reward))
    if total_reward > bipedalWalker.highscore:
        bipedalWalker.highscore = total_reward
    return total_reward



# ========================================= RUN ALGORITHM =========================================

for episode in range(1, MAX_EPISODES + 1):
    if (episode > (MAX_EPISODES / 3) - 200): 
        alpha = 0.5
    if (episode > (MAX_EPISODES * 2 / 3)): alpha = 0.9



    # if(alpha>0.1 and test!="y"):
    #     alpha-=0.001



    episode_score = runEpisode(MAX_TIMESTEPS, bipedalWalker, env, qLearning, episode)
    plotEpisode(myGraph, xval, yval, episode_score, plotLine, episode)



    if(qLearning.epsilon>0.005 and test!="y"):
          qLearning.epsilon-=0.05
  
    # if(gamma<0.9):
    #     gamma+=0.001
    print("epsilon : "+str(qLearning.epsilon))
    print("alpha : "+str(alpha))
    print("gamma : "+str(gamma))

    print("plot_ends \n ________________________________________________________________")
    


#saving qvalues
np.save(save_path, np.array(dict(qLearning.QTable)))
env.close()