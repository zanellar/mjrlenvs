import os 
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mjrlenvs.scripts.pkgpaths import PkgPath

def wrapenv(env,args,load_norm_env=False): 

    env = Monitor(env)                      # A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data
    env = DummyVecEnv([lambda : env])       # Needed for all environments (e.g. used for mulit-processing)
    
    if args.NORMALIZE_ENV is not None:      # BUG we obtain different normalizations for env_expl and env_eval?!

        if load_norm_env:
            normalized_env_path = os.path.join(PkgPath.OUT_TRAIN_FOLDER,f"{args.ENVIRONMENT}/{args.RUN_ID}/normalized_env.pickle") 
            env = VecNormalize.load(normalized_env_path, env)
        else:
            env = VecNormalize(
                env, 
                norm_obs = args.NORMALIZE_ENV["norm_obs"], 
                norm_reward = args.NORMALIZE_ENV["norm_reward"], 
                clip_obs = args.NORMALIZE_ENV["clip_obs"], 
                clip_reward = args.NORMALIZE_ENV["clip_reward"]
            )          

    return env
