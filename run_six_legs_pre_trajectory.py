import gym
import sys
sys.path.append("algs")
sys.path.append("envs")
#from DQN import DeepQNetwork
from DDPG_six_legged import DDPG
from six_legged_env import SixLeggedEnv
import matplotlib.pyplot as plt
import numpy as np
import pickle

plot = 0   #Plot the learning curve?

class DynamicPlot():
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
        plt.ylim((-5, 5))
        plt.title('Learning Curve - six_legged actual robot with DDPG')
        self.figure.canvas.flush_events()

    def close(self):
        plt.close(self.figure)

# Pickle data and save file
def save(args, obs_data, act_data):
    print('Serializing training set ...')
    #make a new file!
    model_file = 'expertPolicy/' + args.name + str(args.rounds) + '.pkl'
    fileObject = open(model_file, 'wb')
    obj = {'obs': obs_data, 'act': act_data}
    pickle.dump(obj, fileObject)
    fileObject.close()
    print('Serialized and saved!')

def run_six_leg(rl_agent):

    step = 0
    qpos0 = env.get_actuator_pos0()
    print(qpos0[:])

    for episode in range(1):
        # initial observation
        observation = env.reset()
        
        if plot:
           d = DynamicPlot()
           iteration = 1

        count = 0
        while True:
            
            # fresh env
            env.render()

            # RL choose action based on observation
            action = expert[step][:]

            print('-----------  action begin  ------------')
            print(action)
            print('-----------  action end  ------------')
            # RL take action and get next observation and reward
            for s in range(25):
                observation, reward, done, info = env.step(action)

            print('-----------  observation begin  ------------')
            print(observation)
            print('-----------  observation end  ------------')

            act_expert.append(action)  #The dimension of action is 18
            obs_expert.append(observation.tolist())   #The dimension of observation is 167

            qpos = env.get_actuator_pos()
            print(qpos[:])
            print("-----------  leg position begin  ------------")
            #print('------------ leg 1 ----------------')
            print("[[%.2f, %.2f, %.2f]," % (qpos[7]/np.pi*180, qpos[8]/np.pi*180, qpos[9]/np.pi*180))
            #print('------------ leg 2 ----------------')
            print("[%.2f, %.2f, %.2f]," % (qpos[10]/np.pi*180, qpos[11]/np.pi*180, qpos[12]/np.pi*180))
            #print('------------ leg 3 ----------------')
            print("[%.2f, %.2f, %.2f]," % (qpos[13]/np.pi*180, qpos[14]/np.pi*180, qpos[15]/np.pi*180))
            #print('------------ leg 4 ----------------')
            print("[%.2f, %.2f, %.2f]," % (qpos[16]/np.pi*180, qpos[17]/np.pi*180, qpos[18]/np.pi*180))
            #print('------------ leg 5 ----------------')
            print("[%.2f, %.2f, %.2f]," % (qpos[19]/np.pi*180, qpos[20]/np.pi*180, qpos[21]/np.pi*180))
            #print('------------ leg 6 ----------------')
            print("[%.2f, %.2f, %.2f]]," % (qpos[22]/np.pi*180, qpos[23]/np.pi*180, qpos[24]/np.pi*180))
            print("-----------  leg position end  ------------")

            if plot:
               # Plot reward here
               d.update(iteration, reward)
               iteration = iteration+1


            # break while loop when end of this episode
            # Note this part is disabled since it cause constantly quitting with error -> need to fix the 'done' judgement
            #if done or step % 5000 == 0:

            #    if plot:
            #       d.close()
            #       d.__init__()
            #       iteration = 1

            #    env.reset()

            step += 1
            if (step == 8):
                step = 0

            count=count+1

            if count==100:
                break

    # end 
    print('over')
    #sys.exit()

class fileName():

    def __init__(self):
        self.name = ''
        self.rounds = 0

if __name__ == "__main__":
    expert = [ \
        [-0.12909389, 0.3230989, -0.3151643, 0.05632893, 0.31895516,
         -0.28769827, 0.04774928, 0.31189033, -0.26120125, -0.12909389,
         0.3230989, -0.3151643, 0.05632893, 0.31895516, -0.28769827,
         0.04774928, 0.31189033, -0.26120125],

        [-0.10478499, 0.38209646, -0.3385682, 0.0283867, 0.31950387,
         -0.29026163, 0.06442661, 0.37072305, -0.30841235, -0.10478499,
         0.32191046, -0.30392185, 0.0283867, 0.37657554, -0.3232424,
         0.06442661, 0.31634586, -0.2768472],

        [-0.08333333, 0.44055556, -0.35722222, -0.05632893, 0.31895516,
         -0.28769827, 0.08333333, 0.44055556, -0.35722222, -0.04774928,
         0.31189033, -0.26120125, 0., 0.44055556, -0.35722222,
         0.12909389, 0.3230989, -0.3151643],

        [-0.06442661, 0.37072305, -0.30841235, -0.08342262, 0.31798558,
         -0.28344571, 0.10478499, 0.38209646, -0.3385682, -0.033006,
         0.30633063, -0.24426278, -0.0283867, 0.37657554, -0.3232424,
         0.15652619, 0.3233781, -0.32475787],

        [-0.04774928, 0.31189033, -0.26120125, -0.10933571, 0.31652358,
         -0.27753331, 0.12909389, 0.3230989, -0.3151643, -0.0199322,
         0.29969919, -0.22609982, -0.05632893, 0.31895516, -0.28769827,
         0.18723439, 0.32295912, -0.33263113],

        [-0.06442661, 0.31634586, -0.2768472, -0.05632893, 0.37548604,
         -0.32039295, 0.10478499, 0.32191046, -0.30392185, -0.04774928,
         0.36350472, -0.29134229, -0.0283867, 0.31950387, -0.29026163,
         0.12909389, 0.38618119, -0.35138642],

        [-0.08333333, 0.31968091, -0.29111806, 0., 0.44055556,
         -0.35722222, 0.08333333, 0.31968091, -0.29111806, -0.08333333,
         0.44055556, -0.35722222, 0., 0.31968091, -0.29111806,
         0.08333333, 0.44055556, -0.35722222],

        [-0.10478499, 0.32191046, -0.30392185, 0.0283867, 0.37657554,
         -0.3232424, 0.06442661, 0.31634586, -0.2768472, -0.10478499,
         0.38209646, -0.3385682, 0.0283867, 0.31950387, -0.29026163,
         0.06442661, 0.37072305, -0.30841235]]

    act_expert = []
    obs_expert = []

    ###get environment
    #env = gym.make('Ant-v2')##HalfCheetah, Ant, Humanoid
    env = SixLeggedEnv()
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
    #rl_agent.restore()
    #parse rl_agent to run the environment

    run_six_leg(rl_agent)
    args = fileName()
    args.name = 'Silvia_expert'
    args.rounds = 100
    save(args, obs_expert, act_expert)
