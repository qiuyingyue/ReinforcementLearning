import gym
import sys
sys.path.append("algs")
sys.path.append("envs")
from DQN import DeepQNetwork
from six_legged_env import SixLeggedEnv
from two_legged_env_dqn import TwoLeggedEnv
import numpy as np
import matplotlib.pyplot as plt

class DynamicPlot():
    #Class used for plotting
    #__init__ for start a new plot
    #update for updating the plot
    #close to close the current plot
    plt.ion()
    def __init__(self):

        #Initialize data
        self.xdata = []
        self.ydata = []

        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], '-')

        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        
        #Other stuff
        self.ax.grid()

    def update(self, iteration, reward):
        self.xdata.append(iteration)
        self.ydata.append(reward)

        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(self.xdata)
        self.lines.set_ydata(self.ydata)

        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()

        #We need to draw *and* flush
        self.figure.canvas.draw()
        plt.xlabel('iteration')
        plt.ylabel('reward')
        plt.xlim((0, iteration+1))
        plt.ylim((-5, 20))
        plt.title('Learning Curve - two_legged with DQN')
        self.figure.canvas.flush_events()

    def close(self):
        plt.close(self.figure)



MAX_EPISODES = 200
MAX_EP_STEPS = 2000
isTrain = True
def run_ant(rl_agent):
    step = 0
    d = DynamicPlot()
    for episode in range(MAX_EPISODES):
        # initial observation
        observation = env.reset()
        print("reset")
        for i in range(MAX_EP_STEPS):    
            # fresh env
            if (not isTrain) or episode >= MAX_EPISODES/6*5:
                env.render()
            # RL choose action based on observation
            action, action_idx = rl_agent.choose_action(observation)
            #action = env.action_space.sample()
            
            # RL take action and get next observation and reward
            
            observation_, reward, done, info = env.step(action)

            # Plot reward here
            d.update(i, reward)

            if isTrain:
                rl_agent.store_transition(observation, action_idx, reward, observation_)

            if isTrain and (step > 200) and (step % 5 == 0):
                rl_agent.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
            #print(info)
            if (step % 300 == 0):
                print("reward:",reward, "info:", info)

        d.close()
        d.__init__()

    # end 
    print('over')
    sys.exit()
if __name__ == "__main__":
    ###get environment
    #env = gym.make('HalfCheetah-v2')##HalfCheetah, Ant, Humanoid
    env = TwoLeggedEnv()#SixLeggedEnv()
    #env = myEnv() #self-defined enviornment


    ###initialize rl_agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(action_dim)
    if (isTrain):
        model_path = "models/dqn_two_legged"
    else:
        model_path = "models/dqn_two_legged_final"
    rl_agent = DeepQNetwork(model_path, action_dim, state_dim,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    #parse rl_agent to run the environment
    run_ant(rl_agent)
    
