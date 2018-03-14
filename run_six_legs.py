

import gym
import sys
sys.path.append("algs")
sys.path.append("envs")
from DQN import DeepQNetwork
from two_legged_env_dqn import TwoLeggedEnv
from four_legged_env_dqn import FourLeggedEnv
from six_legged_env import SixLeggedEnv
import numpy as np
MAX_EPISODES = 200
MAX_EP_STEPS = 200
isTrain = False
def run_ant(rl_agent):
    step = 0
    for episode in range(MAX_EPISODES):
        # initial observation
        observation = env.reset()
        print("reset")

        for i in range(MAX_EP_STEPS):    
            # fresh env
            if (not isTrain) or episode >= MAX_EPISODES - 10:
                env.render()
            # RL choose action based on observation
            #action, action_idx = rl_agent.choose_action(observation)
            action = env.action_space.sample()
            #print(action)
            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

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

    # end 
    print('over')
    sys.exit()
if __name__ == "__main__":
    ###get environment
    #env = gym.make('HalfCheetah-v2')##HalfCheetah, Ant, Humanoid
    env = SixLeggedEnv()
    
    ###initialize rl_agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(action_dim)
    rl_agent = DeepQNetwork("models/dqn_four_legged", action_dim, state_dim,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.7,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    #parse rl_agent to run the environment
    run_ant(rl_agent)
    
