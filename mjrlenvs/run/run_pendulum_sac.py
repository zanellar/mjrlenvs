
from mjrlenvs.scripts.train.trainer import run 
from mjrlenvs.scripts.eval.tester import TestRun
from mjrlenvs.scripts.train.trainutils import linear_schedule 
from mjrlenvs.scripts.args.runargsbase import DefaultArgs
from mjrlenvs.envrl.pendulum import Pendulum 

class Args(DefaultArgs): 

    ############################## RUN #########################################################

    RUN_ID = "prova3"    
    EXPL_EPISODE_HORIZON = 2500 # timesteps 
    EVAL_EPISODE_HORIZON = 500 # timesteps  
    TRAINING_EPISODES = 500 # episodes
    EVAL_MODEL_FREQ = 10*EXPL_EPISODE_HORIZON 
    NUM_EVAL_EPISODES = 5
    NUM_EVAL_EPISODES_BEST_MODEL = 1
    REPETE_TRAINING_TIMES = 20 # times
    SAVE_EVAL_MODEL_WEIGHTS = True 
    SAVE_CHECKPOINTS = True
    EARLY_STOP = False
    EARLY_STOP_MAX_NO_IMPROVEMENTS = 3
    EARLY_STOP_MIN_EVALS = 5
    SAVE_ALL_TRAINING_LOGS = False
 
    ########################## ENVIRONMENT ######################################################

    ENVIRONMENT = "pendulum" 

    ENV_EXPL = Pendulum(
              max_episode_length=EXPL_EPISODE_HORIZON, 
              init_joint_config = "random" 
            )

    ENV_EVAL = Pendulum(
              max_episode_length=EVAL_EPISODE_HORIZON, 
              init_joint_config = "random" 
            )

    NORMALIZE_ENV = dict(training=True, norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=10) 

    ENV_EVAL_RENDERING = False

    ############################## AGENT #######################################################

    AGENT = "SAC"  

    AGENT_PARAMS = dict(
        seed = [17],
        buffer_size = [int(1e6)],
        batch_size = [128],
        learning_starts = [1*EXPL_EPISODE_HORIZON],
        train_freq = [(500,"step") ], 
        gradient_steps = [1000],
        learning_rate = [ linear_schedule(1e-3) ],
        gamma = [0.99],
        tau = [1e-3],
        noise = ["gauss"],
        sigma = [0.1],
        policy_kwargs = [
                        dict(#log_std_init=-2, 
                        net_arch=dict(pi=[256,256],qf=[256,256]))
                        ],
        use_sde_at_warmup = [True ],
        use_sde = [True ],
        sde_sample_freq = [EXPL_EPISODE_HORIZON], 
        ent_coef = ['auto'], 
        target_update_interval = [5],   
    )
 

############################################################################################
############################################################################################
############################################################################################

if __name__ == "__main__":
    x = input("Train[t] or Eval[e]? ")
    if x == "t":
        run(Args())
    elif x == "e":
        tester = TestRun(Args)
        # for name in  
            # mean, std = tester.eval_returns_model(name, n_eval_episodes=5, render=False, save=True)
        tester.plot(show=False)
    else:
        exit(f"Wrong Selection: {x}")