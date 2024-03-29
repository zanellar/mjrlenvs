import os
from traceback import print_tb 
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mjrlenvs.scripts.args.pkgpaths import PkgPath

def wrapenv(env,args,load_norm_env=False): 
    """
    Vectorized (and if args.NORMALIZE_ENV is not None, also normalized) environment wrapper.
    @param env: (gym.Env) The environment to wrap
    @param args: (class) The parsed arguments 
    @param load_norm_env: (bool) Whether to load the normalized environment or not
    @return: (gym.Env) The wrapped environment
    """

    env = Monitor(env)                      # A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data
    env = DummyVecEnv([lambda : env])       # Needed for all environments (e.g. used for mulit-processing)
    
    if args.NORMALIZE_ENV is not None:      

        out_train_folder = PkgPath.OUT_TRAIN_FOLDER if args.OUT_TRAIN_FOLDER is None else args.OUT_TRAIN_FOLDER

        if load_norm_env:
            normalized_env_path = os.path.join(out_train_folder, args.ENVIRONMENT, args.RUN_ID, "normalized_env.pickle") 
            env = VecNormalize.load(normalized_env_path, env)
            print(normalized_env_path)
        else:
            env = VecNormalize(
                env, 
                norm_obs = args.NORMALIZE_ENV["norm_obs"], 
                norm_reward = args.NORMALIZE_ENV["norm_reward"], 
                clip_obs = args.NORMALIZE_ENV["clip_obs"], 
                clip_reward = args.NORMALIZE_ENV["clip_reward"]
            )          

    return env
