
import numpy as np
import math
from gym import spaces

from mjrlenvs.scripts.env.mjenv import MjEnv 
from mjrlenvs.scripts.env.envgymbase import EnvGymBase
 
class Pendulum(EnvGymBase): 

  def __init__(self, 
              max_episode_length=5000, 
              init_joint_config = [0], 
              init_joint_config_std_noise = 0,
              debug = False,
              folder_path = None,
              env_name="pendulum",
              hard_reset = False,
              reward_id = 0,
              ):
    super(Pendulum, self).__init__()
 
    self.debug = debug 
    self.hard_reset = hard_reset
    self.reward_id = reward_id
    
    # Env params
    self.sim = MjEnv( 
      env_name=env_name, 
      folder_path=folder_path,
      max_episode_length=max_episode_length,
      init_joint_config=init_joint_config,
      init_joint_config_std_noise=init_joint_config_std_noise
      ) 
 
    # Actions  
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.sim.action_shape, dtype=np.float32)   

    # Observations 
    self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) 
    
    # Initialize   
    self.action = None
    self.obs = None
    self.reward = None 
    self.done = False
    self.info = {}

  def reset(self, goal=None ):
    """ 
    :return: (np.array) 
    """   
    self.sim.reset(hard_reset=self.hard_reset )
    self.obs = self.get_obs() 
    return self.obs

  def compute_reward(self):   
    sin_pos, cos_pos,  tanh_vel = self.get_obs() 
    err_pos = 1. - sin_pos  
    torque = self.action[0]
    if self.reward_id == 0:
      reward = -abs(err_pos) -0.1*abs(tanh_vel) -0.01*abs(torque)
    elif self.reward_id == 1:
      reward = 1/(1. + abs(err_pos) + 0.1*abs(tanh_vel) + 0.01*abs(torque)) 
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