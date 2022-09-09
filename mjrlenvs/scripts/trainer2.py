 
from ast import arg
from pickle import TRUE
import numpy as np
import os
import itertools  
import pandas as pd
 
from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback,CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise, ActionNoise
from stable_baselines3.common.evaluation import evaluate_policy 
 
from mjrlenvs.scripts.pkgpaths import PkgPath  
from mjrlenvs.scripts.callbacks import Time2EndCallback, SaveTrainingLogsCallback  
from mjrlenvs.scripts.envutils import wrapenv


class SaveTrainingConfigurations():

    def __init__(self, output_file_path, args): 
        self.output_file_path = output_file_path

        with open(self.output_file_path, 'w') as file:
            self.args = args
            line = "agent,"
            for k in self.args.AGENT_PARAMS.keys():
                line += k + ","
            line += "mean,std"  
            file.write(line)
            file.close() 
    
    def add(self, name, config, res):   
        with open(self.output_file_path, 'w') as file:
            line = f"\n{name},"
            for v in config.values():
                line += str(v) + "," 
            line += f"{res[0]},{res[1]}"   
            file.write(line)
            file.close()
  

############################################################################################

def run(args): 

    ##### PRINT RUN ARGUMENTS 
    print("@"*100)
    for k in dir(args):
        if not k.startswith("__"):
            print(f"{k}={getattr(args, k)}") 
    print("@"*100) 


    ################################# ENVIRONMENT ########################################## 
      
    # EXPLORATION ENV  

    env_expl  = args.ENV_EXPL
    env_expl = wrapenv(env_expl,args)

    # EVALUATION ENV 

    env_eval  = args.ENV_EVAL
    env_eval = wrapenv(env_eval,args)

    ############################### CONFIGURATIONS ######################################### 
    
    if not os.path.exists(PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.RUN_ID}/")):
        os.makedirs(PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.RUN_ID}/"))

    # output file with mean reward for each training
    output_file_path = PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.RUN_ID}/"+"run_results.csv")  
    save_training_data = SaveTrainingConfigurations(output_file_path, args)
 
    keys = args.AGENT_PARAMS.keys()
    values = (args.AGENT_PARAMS[key] for key in keys)
    allconfigurations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]



    ############################### TRAININGS #############################################  
    
    for i,config in enumerate(allconfigurations):  
    
        for j in range(args.REPETE_TRAINING_TIMES):

            name = args.AGENT+f"_{i+1}_{j}"
            tensorboard_logs_path = PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.RUN_ID}/tensorboard/")
            save_checkpoints_path = PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.RUN_ID}/checkpoints/{name}/all_cps")
            best_model_folder_path = PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.RUN_ID}/checkpoints/{name}/best_model")
            normalized_env_save_path = PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.RUN_ID}/normalized_env.pickle") 
            save_training_logs_file_path = PkgPath.trainingdata(f"{args.ENVIRONMENT}/{args.RUN_ID}/logs")

            ###################### AGENTS ######################## 

            if config["noise"] is None: 
                action_noise = None 
            else: 
                if config["noise"]=="walk":
                    noise_func = OrnsteinUhlenbeckActionNoise 
                elif config["noise"]=="gauss":
                    noise_func =  NormalActionNoise   
                else:
                    noise_func =  ActionNoise
                action_noise = noise_func(
                    mean = np.zeros(env_expl.action_space.shape), 
                    sigma = config["sigma"]* np.ones(env_expl.action_space.shape)  
                ) 

            ###

            if args.AGENT == "DDPG": 
                agent_class = DDPG
                agent = DDPG(
                    policy = 'MlpPolicy',
                    env = env_expl,  
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
                    tensorboard_log =  tensorboard_logs_path
                ) 

            if args.AGENT == "SAC":
                agent_class = SAC
                agent = SAC(
                    policy = 'MlpPolicy',
                    env = env_expl,  
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
                    tensorboard_log =  tensorboard_logs_path
                ) 

            if args.AGENT == "TD3": 
                agent_class = TD3
                agent = TD3( 
                    policy = 'MlpPolicy',
                    env = env_expl,  
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
                    tensorboard_log = tensorboard_logs_path 
                ) 
        

            ###################### CALLBACKS ######################## 

            callbackslist = [] 

            ###### SAVE TRAINING LOGS  
            callbackslist.append(
                SaveTrainingLogsCallback(
                    folder_path = save_training_logs_file_path,
                    file_name = name,
                    no_rollouts_episode = int(args.EXPL_EPISODE_HORIZON/config["train_freq"][0]),
                    save_all = args.SAVE_ALL_TRAINING_LOGS
                )
            )
            
            ###### PREDICT ENDING TIME   
            callbackslist.append(
                Time2EndCallback(
                    repete_times = args.REPETE_TRAINING_TIMES, 
                    number_configs = len(allconfigurations),
                    number_episodes = args.TRAINING_EPISODES,
                    episode_horizon = args.EXPL_EPISODE_HORIZON,
                    config_index = i,
                    training_index = j
                )
            ) 
  
            ###### SAVE CHECKPOINTS 
            if args.SAVE_CHECKPOINTS: 
                callbackslist.append(
                    CheckpointCallback(
                        save_freq = args.EVAL_MODEL_FREQ, 
                        save_path = save_checkpoints_path
                    )
                )
 
            ###### EARLY STOPPING 
            if args.EARLY_STOP: 
                early_stop_callback = StopTrainingOnNoModelImprovement(
                        max_no_improvement_evals=args.EARLY_STOP_MAX_NO_IMPROVEMENTS, 
                        min_evals=args.EARLY_STOP_MIN_EVALS, 
                        verbose=1) 
            else:
                early_stop_callback = None

            ###### EVALUATION 
            callbackslist.append(
                EvalCallback(
                    env_eval, 
                    best_model_save_path = best_model_folder_path, 
                    eval_freq = args.EVAL_MODEL_FREQ,
                    n_eval_episodes = args.NUM_EVAL_EPISODES, 
                    deterministic = True, 
                    render = args.ENV_EVAL_RENDERING,
                    callback_after_eval = early_stop_callback 
                ) 
            )  # BUG: need to comment "sync_envs_normalization" function in EvalCallback._on_step() method 
             

            callbacks = CallbackList(callbackslist)
        

            ###################### LEARNING ######################## 

            env_expl.reset()   
            agent.learn(
                total_timesteps = args.TRAINING_EPISODES*args.EXPL_EPISODE_HORIZON, 
                log_interval = 1,  
                tb_log_name = args.AGENT+f"_{i+1}_{j}",
                callback = callbacks
            )     

            if args.NORMALIZE_ENV is not None: 
                if not normalized_env_save_path:
                    os.makedirs(normalized_env_save_path)
                env_eval.save(normalized_env_save_path)
 


            ##############################################  

            # Evaluate the best model  
            best_model_file_path = os.path.join(best_model_folder_path,"best_model.zip")
            best_model = agent_class.load(best_model_file_path)  
            mean_reward, std_reward = evaluate_policy(
                                        best_model,  
                                        env_eval, 
                                        n_eval_episodes=args.NUM_EVAL_EPISODES_BEST_MODEL, 
                                        render=False, 
                                        deterministic=True
                                    )   

            save_training_data.add(name, config, res=[mean_reward, std_reward])            
