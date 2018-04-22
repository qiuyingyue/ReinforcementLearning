import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import sys, os
#This file is doing the same thing as gym.make does
class SixLeggedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    x = 0
    y = 0
    t = 0
    def __init__(self):

        #Choose one of the following: first->simple hexapod, second->15dof hexapod, third->full hexapod
        #xml_path = os.path.split(os.path.realpath(__file__))[0]+"/../xmls/six-legged.xml"
        #xml_path = os.path.split(os.path.realpath(__file__))[0]+"/../xmls/six-legged_15dof.xml"
        #xml_path = os.path.split(os.path.realpath(__file__))[0]+"/../xmls/Silvia.xml"
        #xml_path = os.path.split(os.path.realpath(__file__))[0]+"/../xmls/simple_Silvia.xml"
        xml_path = os.path.split(os.path.realpath(__file__))[0]+"/../xmls/simple_Silvia_for_HexaPy.xml"

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)
        self.x = 0
        self.y = 0
        self.t = 0

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        #print ("xposbefore", xposbefore)
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]

        #print ("xposafter", xposafter)
        self.x  = self.x+(xposafter - xposbefore)
        self.y = self.y+(yposafter - yposbefore)
        self.t = self.t + self.dt
        forward_reward = self.x/self.t 
        forward_cost = self.y/self.t

        ctrl_cost = .4 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - abs(forward_cost) - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            forward_cost=abs(forward_cost),
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
    
    def get_actuator_pos(self):
        #Get mjModel by sim.model, mjData by sim.data
        return self.sim.data.qpos.flat

    def get_actuator_pos0(self):
        return self.sim.model.qpos0.flat

    def get_actuator_pos(self):
        return self.sim.data.qpos.flat

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
