from pickle import TRUE
import numpy as np
import os
import itertools  
import time
 
from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, EvalCallback,CheckpointCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy 
 
from mjrlenvs.scripts.pkgpaths import PkgPath 
 

def run(args): 
    
    ############################################################################################
    ########################## ENVIRONMENT #####################################################
    ############################################################################################
     
    env  = args.ENV
 
    env = Monitor(env)                      # A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data
    env = DummyVecEnv([lambda : env])     # Needed for all environments (e.g. used for mulit-processing)
    env = VecNormalize(env)               # Needed for improving training when using MuJoCo envs?

    ############################################################################################
    ################################### AGENT ##################################################
    ############################################################################################
    
    if not os.path.exists(PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.TRAINING_ID}/")):
        os.makedirs(PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.TRAINING_ID}/"))

    output_file_path = PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.TRAINING_ID}/"+"output.txt")

    with open(output_file_path, 'w') as f: 
        line = "agent,"
        for k in args.AGENT_PARAMS.keys():
            line += k + ","
        line += "mean,std" 
        line += ",hours2end" 
        f.write(line)
        f.close()

    ############################################################################################

    keys = args.AGENT_PARAMS.keys()
    values = (args.AGENT_PARAMS[key] for key in keys)
    allconfigurations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    ############################################################################################
    
    for i,config in enumerate(allconfigurations):  
    
        for j in range(args.REPETE_TRAINING_TIMES):

            t1 = time.time() 

            if config["noise"] is None: 
                action_noise = None 
            else: 
                if config["noise"]=="ou":
                    noise_func = OrnsteinUhlenbeckActionNoise 
                else:
                    noise_func =  NormalActionNoise   
                action_noise = noise_func(
                    mean = np.zeros(env.action_space.shape), 
                    sigma = config["sigma"]* np.ones(env.action_space.shape)  
                ) 


            if args.AGENT == "DDPG": 
                agent_class = DDPG
                agent = DDPG(
                    policy = 'MlpPolicy',
                    env = env,  
                    buffer_size = config["buffer_size"],
                    batch_size = config["batch_size"], 
                    learning_starts = config["learning_starts"], 
                    train_freq = config["train_freq"], 
                    gradient_steps =  config["gradient_steps"],  
                    learning_rate = config["learning_rate"], 
                    gamma = config["gamma"],
                    tau = config["tau"],     
                    action_noise = action_noise, 
                    policy_kwargs = config["policy_kwargs"],
                    verbose = 1, 
                    tensorboard_log =  PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.TRAINING_ID}/tensorboard/")
                ) 

            if args.AGENT == "SAC":
                agent_class = SAC
                agent = SAC(
                    policy = 'MlpPolicy',
                    env = env,  
                    buffer_size = config["buffer_size"],
                    batch_size = config["batch_size"], 
                    learning_rate = config["learning_rate"], 
                    gamma = config["gamma"],
                    tau = config["tau"],  
                    action_noise = action_noise,
                    learning_starts = config["learning_starts"], 
                    train_freq = config["train_freq"],   
                    gradient_steps =  config["gradient_steps"],  
                    seed = config["seed"], 
                    #
                    use_sde_at_warmup = config["use_sde_at_warmup"],
                    use_sde = config["use_sde"], 
                    sde_sample_freq = config["sde_sample_freq"],
                    target_update_interval = config["target_update_interval"], 
                    ent_coef = config["ent_coef"],  
                    policy_kwargs = config["policy_kwargs"],
                    #
                    verbose = 1, 
                    tensorboard_log =  PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.TRAINING_ID}/tensorboard/")
                ) 

            if args.AGENT == "TD3": 
                agent_class = TD3
                agent = TD3( 
                    policy = 'MlpPolicy',
                    env = env,  
                    buffer_size = config["buffer_size"],
                    batch_size = config["batch_size"], 
                    learning_starts = config["learning_starts"], 
                    train_freq = config["train_freq"], 
                    gradient_steps =  config["gradient_steps"],  
                    learning_rate = config["learning_rate"],  
                    gamma = config["gamma"],
                    tau = config["tau"],  
                    action_noise = action_noise,
                    policy_kwargs = config["policy_kwargs"],
                    seed = config["seed"], 
                    #
                    policy_delay = config["policy_delay"], 
                    target_policy_noise =  config["target_policy_noise"], 
                    target_noise_clip =  config["target_noise_clip"],
                    #
                    verbose = 1, 
                    tensorboard_log =  PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.TRAINING_ID}/tensorboard/")
                ) 
        
            ######################################################################## 

            callbackslist = [] 

            if args.SAVE_EVAL_MODEL_WEIGHTS: 
                callbackslist.append(
                    EvalCallback(
                        env, 
                        best_model_save_path = PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.TRAINING_ID}/checkpoints/"+args.AGENT+f"_{i+1}_{j}"+"/best_model"),
                        # log_path = PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.TRAINING_ID}/checkpoints/"+args.AGENT+f"_{i+1}_{j}"+"/eval_results"), 
                        eval_freq = args.EVAL_MODEL_FREQ,
                        n_eval_episodes = args.NUM_EVAL_EPISODES, 
                        deterministic = True, 
                        render = True
                    )
                )  # NB: need to comment "sync_envs_normalization" function in EvalCallback._on_step() method 
        
            callbacks = CallbackList(callbackslist)
        
            ######################################################################## 

            env.reset()   
            agent.learn(
                total_timesteps = args.TRAINING_EPISODES*args.EPISODE_HORIZON, 
                log_interval = 1,  
                tb_log_name = args.AGENT+f"_{i+1}_{j}",
                callback = callbacks
            )  

            env.close()
    
            ########################################################################  

            best_model = agent_class.load(PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.TRAINING_ID}/checkpoints/"+args.AGENT+f"_{i+1}_{j}"+"/best_model/best_model.zip"))
            mean_reward, std_reward = evaluate_policy(
                                        best_model,  
                                        env, 
                                        n_eval_episodes=args.NUM_EVAL_EPISODES_BEST_MODEL, 
                                        render=False, 
                                        deterministic=True
                                    )

            t2 = time.time()
            hours2end = ((t2-t1)*(len(allconfigurations)*args.REPETE_TRAINING_TIMES - i*args.REPETE_TRAINING_TIMES - j)/60)/60
            
            with open(output_file_path, 'a') as f:  
                line = f"\n{args.AGENT}_{i+1}_{j},"
                for v in config.values():
                    line += str(v) + ","
                line += f"{mean_reward},{std_reward}"  
                line += f",{hours2end}" 
                f.write(line)
                f.close()
            