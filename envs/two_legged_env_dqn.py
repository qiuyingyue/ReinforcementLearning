import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env

class TwoLeggedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = os.path.split(os.path.realpath(__file__))[0]+"/../xmls/two-legged.xml"
        
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        #print("self.sim.data.cfrc_ext",self.sim.data.cfrc_ext)
        body_height = self.get_body_com("head")[2]
        #print("head",pos_head)
        contact_top = self.sim.data.cfrc_ext[1] + self.sim.data.cfrc_ext[2]
        cost_contact =  np.square(np.clip(contact_top, -1, 1)).mean()
        cost_jump = body_height - 0.8 if body_height > 0.8 else 0#incase it jumps too high
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()#0.1
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run - cost_contact - cost_jump
        if (body_height < 0.3):
            done = True
        else:
            done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, reward_contact = -cost_contact, reward_jump = -cost_jump)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
