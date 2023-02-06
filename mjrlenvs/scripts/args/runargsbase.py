
from mjrlenvs.scripts.train.trainutils import linear_schedule 
from mjrlenvs.scripts.env.envgymbase import EnvGymBase  

class DefaultArgs():
    '''
    This class contains the arguments to be used for training and testing the model.
    '''

    ############################## RUN #########################################################

    RUN_ID = "default"                              # unique run id
    OUT_TRAIN_FOLDER = None                         # output folder for training logs
    OUT_TEST_FOLDER = None                          # output folder for testing logs
    EXPL_EPISODE_HORIZON = 2500                     # number of timesteps in each exploration episode
    EVAL_EPISODE_HORIZON = 500                      # number of timesteps in each evaluation episode  
    TRAINING_EPISODES = 500                         # number of episodes to train the model
    EVAL_MODEL_FREQ = int(TRAINING_EPISODES/100)    # evaluate the model every EVAL_MODEL_FREQ episodes (default: 10% of training episodes)
    NUM_EVAL_EPISODES = 1                           # number of episodes to evaluate the model
    NUM_EVAL_EPISODES_BEST_MODEL = 1                # number of episodes to evaluate the best model
    REPETE_TRAINING_TIMES = 1                       # number of times to repeat the training process (if some randomness is used)
    SAVE_EVAL_MODEL_WEIGHTS = True                  # save the weights of the best model
    SAVE_CHECKPOINTS = True                         # save the model weights every EVAL_MODEL_FREQ episodes
    EARLY_STOP = True                               # stop training if the model does not improve for EARLY_STOP_MAX_NO_IMPROVEMENTS episodes
    EARLY_STOP_MAX_NO_IMPROVEMENTS = 3              # number of episodes without improvement to stop training
    EARLY_STOP_MIN_EVALS = 5                        # minimum number of evaluations to stop training
    SAVE_ALL_TRAINING_LOGS = False                  # save all training logs (default: False) N.B. this can be very memory consuming, it creates a very huge file!!

    CALLBACKS = []                                  # list of callbacks to use during training (see https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html)
 
    ########################## ENVIRONMENT ######################################################

    ENVIRONMENT = "base"                            # environment name
    ENV_EXPL = EnvGymBase( )                        # environment for exploration. User can pass a custom environment here (from mjrlenvs/envrl)
    ENV_EVAL = EnvGymBase( )                        # environment for evaluation. User can pass a custom environment here (from mjrlenvs/envrl)
    NORMALIZE_ENV = None                            # normalize the environment, in case set "dict(training=True, norm_obs=True, norm_reward=True, clip_obs=1, clip_reward=1)" 
    ENV_EVAL_RENDERING = False                      # render the environment during evaluation
    
    ############################## AGENT #######################################################

    AGENT = "DDPG"                                  # agent name

    AGENT_PARAMS = dict(                            # agent parameters (use list to perform grid search)
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
 
 