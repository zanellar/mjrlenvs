
from mjrlenvs.scripts.trainer import run 
from mjrlenvs.scripts.trainutils import linear_schedule 
from mjrlenvs.envrl.pendulum_ebud import PendulumEBud

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

    ENV = PendulumEBud(
            max_episode_length=EPISODE_HORIZON, 
            energy_tank_init = 1, # initial energy in the tank
            energy_tank_threshold = 0, # minimum energy in the tank  
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
        learning_rate = [linear_schedule(1e-3), 3e-4 ],
        gamma = [0.99],
        tau = [1e-4,1e-3],
        noise = ["ou", None],
        sigma = [0.3,0.8],
        policy_kwargs = [
                        None, 
                        dict(log_std_init=-2, 
                        net_arch=dict(pi=[256,256],qf=[256, 256])),
                        ],
        use_sde_at_warmup = [False,True],
        use_sde = [False,True],
        sde_sample_freq = [EPISODE_HORIZON], 
        ent_coef = ['auto'], 
        target_update_interval = [5],   
    )
 

############################################################################################
############################################################################################
############################################################################################

if __name__ == "__main__":
    run(Args())