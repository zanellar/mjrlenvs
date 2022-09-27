import sys
sys.path.append('/home/kvn/Super Duper Code/panda_mujoco/')
 
from mjrlenvs.envrl.pendulum import Pendulum

env = Pendulum(
  max_episode_length=500, 
  init_joint_config = [-1.57]
) 

obs = env.reset()
for i in range(10000):
    action = [1]
    obs, reward, done, info = env.step(action)
    env.render() 
    print(f"obs={obs}, reward={reward}, done={done}")
    if done: 
      obs = env.reset() 
      
env.close()
