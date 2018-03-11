

import gym
import sys
from DQN import DeepQNetwork
from six_legged_env import SixLeggedEnv
import numpy as np
MAX_EPISODES = 20
MAX_EP_STEPS = 2000
def run_ant(rl_agent):
    step = 0
    for episode in range(MAX_EPISODES):
        # initial observation
        observation = env.reset()
        print("reset")

        for i in range(MAX_EP_STEPS):    
            # fresh env
            if (episode >= MAX_EPISODES - 1):
                env.render()
            # RL choose action based on observation
            action, action_idx = rl_agent.choose_action(observation)
            #action = env.action_space.sample()
            
            
            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)
            #print(action)
            rl_agent.store_transition(observation, action_idx, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                rl_agent.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

            if (step % 300 == 0):
                print("reward:",reward, "info:", info)

    # end 
    print('over')
    sys.exit()
if __name__ == "__main__":
    ###get environment
    env = gym.make('HalfCheetah-v2')##HalfCheetah, Ant, Humanoid
    #env = SixLeggedEnv()
    #env = myEnv() #self-defined enviornment


    ###initialize rl_agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(action_dim)
    rl_agent = DeepQNetwork( action_dim, state_dim,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    #parse rl_agent to run the environment
    run_ant(rl_agent)
    
