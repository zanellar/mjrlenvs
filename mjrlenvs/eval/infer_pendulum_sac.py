import os 
from pickle import FALSE
import numpy as np 
from stable_baselines3 import HER, SAC, TD3, DDPG 
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mjrlenvs.scripts.pkgpaths import PkgPath
from mjrlenvs.envrl.pendulum import Pendulum


AGENT_WEIGHTS_PATH = "/home/riccardo/projects/mjrlenvs/data/train/pendulum/prova2/checkpoints/SAC_1_0/best_model/best_model.zip"
EPISODE_HORIZON = 500 # timesteps 
RENDERING = True  
NUM_EVAL_EPISODES = 1 

########################################################################
  
env = Pendulum(
            max_episode_length=EPISODE_HORIZON, 
            init_joint_config = [0], 
            debug=True
        )
 
env = Monitor(env)                      # A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data
env = DummyVecEnv([lambda : env])     # Needed for all environments (e.g. used for mulit-processing)
# env = VecNormalize(env)               # Needed for improving training when using MuJoCo envs? 

########################################################################

model = SAC.load(AGENT_WEIGHTS_PATH) 
obs = env.reset()

while True:
    print("@@@@@@")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render("frontview")
    if done:
        obs = env.reset()
 
