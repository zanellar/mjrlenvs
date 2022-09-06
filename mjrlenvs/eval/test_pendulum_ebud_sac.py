import os 
from pickle import FALSE
import numpy as np 
from stable_baselines3 import HER, SAC, TD3, DDPG 
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mjrlenvs.scripts.pkgpaths import PkgPath
from mjrlenvs.envrl.pendulum_ebud import PendulumEBud
from mjrlenvs.envrl.pendulum import Pendulum


AGENT_WEIGHTS_PATH = "/home/riccardo/projects/mjrlenvs/data/train/pendulum/prova/checkpoints/SAC_1_0/best_model/best_model.zip"
EPISODE_HORIZON = 500 # timesteps 
RENDERING = True  
NUM_EVAL_EPISODES = 10 

########################################################################
########################################################################
########################################################################

########################################################################

if not os.path.exists(PkgPath.testingdata()):
    os.makedirs(PkgPath.testingdata())

with open(PkgPath.testingdata("output.txt"), 'w') as f: 
    line = "timestep,energy_tank,energy_exchanged" 
    f.write(line)
    f.close()

########################################################################

def energycallback(locals_, _globals): 
    energy_tank = locals_["info"]["energy_tank"]
    print(energy_tank)
    energy_exchanged = locals_["info"]["energy_exchanged"]
    t = locals_["current_lengths"][0]
    with open(PkgPath.testingdata("output.txt"), 'a') as f: 
        line = f"\n{t},{energy_tank},{energy_exchanged}" 
        f.write(line)
        f.close()
 

########################################################################

# env = PendulumEBud(
#         max_episode_length=EPISODE_HORIZON, 
#         energy_tank_init = 10, # initial energy in the tank
#         energy_tank_threshold = 0, # minimum energy in the tank  
#         )

env = Pendulum(
            max_episode_length=EPISODE_HORIZON, 
            init_joint_config = [0]
        )
 
env = Monitor(env)                      # A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data
env = DummyVecEnv([lambda : env])     # Needed for all environments (e.g. used for mulit-processing)
env = VecNormalize(env)               # Needed for improving training when using MuJoCo envs? 

########################################################################


model = SAC.load(AGENT_WEIGHTS_PATH)
 
mean_reward, std_reward = evaluate_policy(
                            model,
                            env, 
                            n_eval_episodes=NUM_EVAL_EPISODES,  
                            render = RENDERING, 
                            deterministic=True,
                            # callback=energycallback
                            )

print(mean_reward, std_reward)

env.close()
