import gym
import sys
sys.path.append("example")
from DQN import DeepQNetwork
from two_legged_env import TwoLeggedEnv
def run_ant(rl_agent):
    step = 0
    for episode in range(10):
        # initial observation
        observation = env.reset()

        while True:
            
            # fresh env
            env.render()
            # RL choose action based on observation
            action = rl_agent.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            rl_agent.store_transition(observation, action, reward, observation_)

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
    #env = gym.make('Ant-v2')##HalfCheetah, Ant, Humanoid
    env = TwoLeggedEnv()
    #env = myEnv() #self-defined enviornment


    ###initialize rl_agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    rl_agent = DeepQNetwork(action_dim, state_dim,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    #parse rl_agent to run the environment
    run_ant(rl_agent)
