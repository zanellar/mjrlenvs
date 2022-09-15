import math
import numpy as np
from gym import spaces
from mjrlenvs.scripts.env.mjenv import MjEnv 
from mjrlenvs.scripts.env.envgymbase import EnvGymBase

class PandaPositionReach(EnvGymBase): 
  
  TARGET_RANGE = [[ 0.3, 0.6], [-0.5, 0.5], [0.3, 0.8]]
  WORKSPACE_BARRIERS = [[0.0, 0.9], [-0.9, 0.9], [0.2, 1.0]]
  SUCCESS_THRESHOLD = 0.05

  def __init__(self, 
              max_episode_length=5000, 
              init_joint_config = 'random',
              debug = False
              ):
    super(PandaPositionReach, self).__init__()
    
    self.debug = debug 
    
    # Env params
    self.sim = MjEnv(
      env_name="panda_position",   
      max_episode_length=max_episode_length,
      init_joint_config=init_joint_config 
    )

    self.num_successful_episodes = 0
    self.reward = 0 
    self.action = None

    # Goal  
    self.goal = self.sim.get_obj_pos("target_point")
    self.achieved_goal = self.sim.get_obj_pos("end_effector")  

    # Actions  
    self.action_space = spaces.Box(low=-1, high=1., shape=self.sim.action_shape, dtype=np.float32)   

    # Observation space  
    self.observation_space =  spaces.Box(low=0, high=1, shape=self.sim.state_shape, dtype=np.float32)

    self.reset()


  def _terminate_episode(self):
    target_pos = self.sim.get_obj_pos("target_point")
    eef_pos = self.sim.get_obj_pos("end_effector")  

    # the gripper goes out of the desired work space
    for i in range(len(eef_pos)):
      minv = self.WORKSPACE_BARRIERS[i][0]
      maxv = self.WORKSPACE_BARRIERS[i][1]
      if eef_pos[i]<minv or eef_pos[i]>maxv:
        print(f"dim {i}:  {eef_pos[i]} out of [{minv},{maxv}]")
        # exit("@@@@@@")
        return True

    # reaches the target
    # if np.linalg.norm(eef_pos-target_pos, axis = -1) < self.SUCCESS_THRESHOLD:
    #   self.num_successful_episodes += 1
    #   #print(f"{np.linalg.norm(eef_pos-target_pos, axis = -1)}<{self.SUCCESS_THRESHOLD}")
    #   return True 

    # the robot hits something
    # TODO

    return False
 
  def compute_reward(self, info): 
    target_pos = self.sim.get_obj_pos("target_point")
    eef_pos = self.sim.get_obj_pos("end_effector")  
    dist = np.linalg.norm(eef_pos-target_pos, axis = -1)   
    r = 1/(dist**2+0.01)
    # r = -dist 
    #print(r)
    return r

  def set_goal(self, goal):
    self.sim.set_site_pos("target_point",goal)
    self.goal = self.sim.get_obj_pos("target_point")

  def reset(self, hard_reset=True, random_goal=True): 
    if random_goal:
      new_goal = np.random.rand(len(self.goal))
      trange = self.TARGET_RANGE
      for i, r in enumerate(trange):  
        new_goal[i] = r[0] + (new_goal[i])*(r[1]-r[0])
      self.set_goal(new_goal)
    self.sim.reset(hard_reset=hard_reset) 
    self.obs = self.get_obs()
    return self.obs
 
  def step(self, action):  
    info = {}   
    self.action = action  
    _, done = self.sim.execute(self.action)    
    self.reward = self.compute_reward(info) 
    self.obs = self.get_obs()
    if not done: 
      done = self._terminate_episode() 

    return self.obs, self.reward, done, info

  def render(self, mode=None): 
    self.sim.render()
    pass 

  def get_obs(self): 
    qpos, qvel = self.sim.get_state()
    self.obs = np.array([
      math.sin(qpos),
      math.cos(qpos),
      math.tanh(qvel)
    ])  
    return self.obs