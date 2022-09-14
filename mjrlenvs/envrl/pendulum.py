from os import PRIO_PGRP
from traceback import print_tb
import numpy as np
import math
from gym import spaces
import matplotlib.pyplot as plt

from mjrlenvs.scripts.pkgpaths import PkgPath
from mjrlenvs.scripts.mjenv import MjEnv 
from mjrlenvs.envrl.base import EnvGymBase
 
class Pendulum(EnvGymBase): 

  def __init__(self, 
              max_episode_length=5000, 
              init_joint_config = [0],
              debug = False
              ):
    super(Pendulum, self).__init__()


    self.debug = debug 
    
    # Env params
    self.sim = MjEnv( 
      env_name="pendulum",   
      max_episode_length=max_episode_length,
      init_joint_config=init_joint_config 
      )

    # Initialize   
    self.action = np.zeros(self.sim.action_shape)
    self.obs = np.zeros(3)
    self.reward = None 
    self.done = False
    self.info = {}
 
    # Actions  
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.sim.action_shape, dtype=np.float32)   

    # Observations 
    self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) 
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32) 

  def reset(self, hard_reset=True, goal=None ):
    """ 
    :return: (np.array) 
    """   
    self.sim.reset(hard_reset=hard_reset )
    self.obs = self.get_obs() 
    return self.obs

  def compute_reward(self):   
    sin_pos, cos_pos,  tanh_vel = self.get_obs() 
    err_pos = abs(1. - sin_pos)
    # reward = 1/(1. + 5*err_pos + tanh_vel)
    torque = self.action[0]
    reward = -err_pos -0.1*abs(tanh_vel) -0.01*abs(torque)
    # print(reward)
    return reward

  def step(self, action):   
    self.action = action  
    _, done = self.sim.execute(self.action) 
    self.reward = self.compute_reward( )  
    self.obs = self.get_obs()
    self.done = done
    self.info = {}
 
    return self.obs, self.reward, self.done, self.info

  def get_obs(self): 
    qpos, qvel = self.sim.get_state()
    self.obs = np.array([
      math.sin(qpos),
      math.cos(qpos),
      math.tanh(qvel)
    ])   
    return self.obs

  def render(self, mode=None): 
    self.sim.render()

  def get_sample(self):
    return self.obs, self.action, self.reward, self.done, self.info