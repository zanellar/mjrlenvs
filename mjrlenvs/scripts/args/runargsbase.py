
from mjrlenvs.scripts.train.trainutils import linear_schedule 
from mjrlenvs.scripts.env.envgymbase import EnvGymBase  

class DefaultArgs(): 

    ############################## RUN #########################################################

    RUN_ID = "default"   
    OUT_TRAIN_FOLDER = None
    OUT_TEST_FOLDER = None
    EXPL_EPISODE_HORIZON = 2500 # timesteps 
    EVAL_EPISODE_HORIZON = 500 # timesteps  
    TRAINING_EPISODES = 500 # episodes
    EVAL_MODEL_FREQ = int(TRAINING_EPISODES/100)
    NUM_EVAL_EPISODES = 1
    NUM_EVAL_EPISODES_BEST_MODEL = 1
    REPETE_TRAINING_TIMES = 1 # times
    SAVE_EVAL_MODEL_WEIGHTS = True 
    SAVE_CHECKPOINTS = True 
    EARLY_STOP = True
    EARLY_STOP_MAX_NO_IMPROVEMENTS = 3
    EARLY_STOP_MIN_EVALS = 5
    SAVE_ALL_TRAINING_LOGS = False # creates a very huge file!!

    CALLBACKS = []
 
    ########################## ENVIRONMENT ######################################################

    ENVIRONMENT = "base"  
    ENV_EXPL = EnvGymBase( )
    ENV_EVAL = EnvGymBase( )
    NORMALIZE_ENV = None # or dict(training=True, norm_obs=True, norm_reward=True, clip_obs=1, clip_reward=1) 
    ENV_EVAL_RENDERING = False
    
    ############################## AGENT #######################################################

    AGENT = "DDPG"  

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
                        ] 
    )
 
 