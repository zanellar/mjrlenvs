
from mjrlenvs.scripts.trainer import run 
from mjrlenvs.scripts.trainutils import linear_schedule 
from mjrlenvs.envrl.pendulum import Pendulum 

class Args():
        

    ############################## RUN #########################################################

    EPISODE_HORIZON = 500 # timesteps 
    TRAINING_EPISODES = 1000 # episodes
    EVAL_MODEL_FREQ = int(TRAINING_EPISODES*EPISODE_HORIZON/10)
    NUM_EVAL_EPISODES = 1
    NUM_EVAL_EPISODES_BEST_MODEL = 1
    REPETE_TRAINING_TIMES = 5 # times
    SAVE_EVAL_MODEL_WEIGHTS = True 
    TRAINING_ID = "1"   
 
    ########################## ENVIRONMENT ######################################################

    ENVIRONMENT = "pendulum" 

    ENV = Pendulum(
              max_episode_length=EPISODE_HORIZON, 
              init_joint_config = [0]
            )

    ############################## AGENT #######################################################

    AGENT = "SAC"  

    AGENT_PARAMS = dict(
        seed = [None,17],
        buffer_size = [int(1e6)],
        batch_size = [128],
        learning_starts = [20*EPISODE_HORIZON],
        train_freq = [(1,"episode")], 
        gradient_steps = [-1 ],
        learning_rate = [linear_schedule(1e-3) ],
        gamma = [0.99],
        tau = [1e-4 ],
        noise = ["ou", None],
        sigma = [0.8],
        policy_kwargs = [ 
                        dict(
                            log_std_init=-2, 
                            net_arch=dict(pi=[256,256],qf=[256, 256])),
                        ],
        use_sde_at_warmup = [ True],
        use_sde = [ True],
        sde_sample_freq = [EPISODE_HORIZON], 
        ent_coef = ['auto'], 
        target_update_interval = [5],   
    )
 

############################################################################################
############################################################################################
############################################################################################

if __name__ == "__main__":
    run(Args())