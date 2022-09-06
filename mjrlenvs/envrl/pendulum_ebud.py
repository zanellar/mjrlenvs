from cmath import inf
from faulthandler import is_enabled
from pickle import TRUE
import numpy as np
import math 
import gym
from mjrlenvs.envrl.pendulum import Pendulum  
 
class PendulumEBud(gym.Env): 

    def __init__(self,    
                max_episode_length = 50, 
                energy_tank_init = 5, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                debug = False,
                testing_mode = False 
                ):
        super(PendulumEBud, self).__init__()

        self.testing_mode = testing_mode
        self.debug = debug
        self.t = 0
        self.T = max_episode_length

        ################# Init Learning Environment ####################

        # Vanilla Environment
        self._env = Pendulum(max_episode_length=max_episode_length)   
 
        # Observations 
        self.observation_space = self._env.observation_space 

        # Actions  
        self.action_space = self._env.action_space 

        # Vanilla obs
        self.obs = self._env.get_obs()  
        self.action = np.zeros(self._env.action_space.shape)

        ################# Init Energy-budgeting Framework ####################
 
        self.energy_tank_init = energy_tank_init
        self.energy_tank = energy_tank_init  
        self.energy_tank_threshold = energy_tank_threshold
        self.energy_avaiable = True 

        self.energy_joints = [0,0] # list containing the energy going out from the 2 motors
        self.energy_exchanged = 0 # initializing total energy going out at a time step to 0 
  
        self.energy_stop_ct = 0

        self.freq = self._env.sim.control_frequency
    
    def _update_energy_tank(self, obs_new):
        ''' 
        Etank(t) = Etank(t-1) - Eex(t)
        Ex(t) = sum(tau(t-1,i)*(q(t,i) - q(t-1,i)), i=1,2)
        '''
        
        q0 = np.arcsin(self.obs[0])
        q1 = np.arcsin(obs_new[0])   
        delta_q = q1 - q0 
 
        tau = self.action
        self.energy_joints = tau*delta_q  
        self.energy_exchanged = sum(self.energy_joints)
        self.energy_tank -= self.energy_exchanged
        
        tank_is_empty = self.energy_tank <= self.energy_tank_threshold
        self.energy_avaiable = not tank_is_empty

        if self.debug:
            print(f"etank={self.energy_tank}, dq={delta_q}, tau={tau}") 

        return tank_is_empty

    def step(self, action):  
  
        if not self.energy_avaiable: 
            action *= 0  

        # Vanilla Environment Step
        obs_new, _reward, _done, _info = self._env.step(action) 

        # Energy Budgeting  
        tank_is_empty = self._update_energy_tank(obs_new) 
  
        horizon_done = self.t>=self.T

        if self.testing_mode: 
            done = _done or horizon_done
        else:
            done = tank_is_empty or _done or horizon_done
 
        if tank_is_empty:
            self.energy_stop_ct += 1
         
        info = dict(energy_exchanged = self.energy_exchanged,
                    energy_tank = self.energy_tank, 
                    _info = _info)   

        self.action = action
        self.obs = obs_new 
        self.t += 1 

        # print(self.t, self.energy_tank, action,  self.energy_joints) 

        return obs_new, _reward, done, info

    def reset(self, hard_reset=True, goal=None):  
        print("@"*100) 
        self.obs = self._env.reset(hard_reset=hard_reset) 
        self.action = np.zeros(self._env.action_space.shape)
        self.t = 0  
        self.energy_tank = self.energy_tank_init
        self.energy_exchanged = 0.0  
        return self.obs 

    def render(self, mode=None): 
        self._env.render()

    def close(self):
        self._env.close()
          
    def seed(self, seed=None):
        return self._env.seed(seed)

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     return self._env.compute_reward(achieved_goal, desired_goal, info)