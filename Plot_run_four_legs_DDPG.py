import gym
import sys
sys.path.append("algs")
sys.path.append("envs")
#from DQN import DeepQNetwork
from DDPG_four_legged import DDPG
from four_legged_env_DDPG import FourLeggedEnv
import matplotlib.pyplot as plt
import numpy as np

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
        plt.title('Learning Curve - four_legged with DDPG')
        self.figure.canvas.flush_events()

    def close(self):
        plt.close(self.figure)


def run_four_leg(rl_agent):
    step = 0
    for episode in range(10):
        # initial observation
        observation = env.reset()

        d = DynamicPlot()
        iteration = 1

        while True:
            
            # fresh env
            env.render()
            # RL choose action based on observation
            action = rl_agent.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            # Plot reward here
            d.update(iteration, reward)
            iteration = iteration+1

            rl_agent.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                rl_agent.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done or step % 5000 == 0:
                
                d.close()
                d.__init__()
                iteration = 1

                env.reset()
            step += 1

            if (step % 300 == 0):
                print("reward:",reward, "info:", info)
            if (step % 2000 == 0):
                rl_agent.save()

    # end 
    print('over')
    sys.exit()
if __name__ == "__main__":
    ###get environment
    #env = gym.make('Ant-v2')##HalfCheetah, Ant, Humanoid
    env = FourLeggedEnv()
    #env = myEnv() #self-defined enviornment


    ###initialize rl_agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    '''rl_agent = DeepQNetwork(action_dim, state_dim,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    '''
    rl_agent = DDPG(action_dim, state_dim, a_bound = (-1, 1))
    rl_agent.restore()
    #parse rl_agent to run the environment
    run_four_leg(rl_agent)
