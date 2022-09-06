 
import gym 

class EnvGymBase(gym.Env): 

  def __init__(self,  
              debug = False
              ):
    super(EnvGymBase, self).__init__()

    self.obs = None
    self.action = None
    self.reward = None
 

  def reset(self, goal=None): 
    pass
  
  def set_goal(self, goal):
    pass
    
  def compute_reward(self):  
    pass

  def step(self, action):  
    pass

  def render(self, mode=None): 
    pass

  def close(self):
    pass

  def get_obs(self):
    pass
 
  def seed(self, seed=None):
    pass

  def get_sample(self):
    return self.obs, self.action, self.reward


class EnvGymGoalBase(EnvGymBase): 

  def __init__(self,  
              debug = False
              ):
    super(EnvGymGoalBase, self).__init__()