
from mjrlenvs.scripts.trainer import run 
from mjrlenvs.scripts.trainutils import linear_schedule 
from mjrlenvs.envrl.pendulum import Pendulum 

class Args():
         
    ############################## RUN #########################################################

    EPISODE_HORIZON = 500 # timesteps 
    TRAINING_EPISODES = 2000 # episodes
    EVAL_MODEL_FREQ = int(TRAINING_EPISODES*EPISODE_HORIZON/10)
    NUM_EVAL_EPISODES = 1
    NUM_EVAL_EPISODES_BEST_MODEL = 1
    REPETE_TRAINING_TIMES = 10 # times
    SAVE_EVAL_MODEL_WEIGHTS = True 
    TRAINING_ID = "1"   
 

    ########################## ENVIRONMENT ######################################################

    ENVIRONMENT = "pendulum" 

    ENV = Pendulum(
              max_episode_length=EPISODE_HORIZON, 
              init_joint_config = [0]
            )

            
    ############################## AGENT #######################################################

    AGENT = "DDPG"  

    AGENT_PARAMS = dict(
        seed = [17],
        buffer_size = [int(1e6)],
        batch_size = [128],
        learning_starts = [5*EPISODE_HORIZON],
        train_freq = [(1,"episode")], 
        gradient_steps = [-1 ],
        learning_rate = [linear_schedule(1e-4), linear_schedule(1e-3)],
        gamma = [0.99],
        tau = [5e-4, 5e-3],
        noise = ["ou"],
        sigma = [0.3],
        policy_kwargs = [None, 
                        dict(log_std_init=-2, 
                        net_arch=dict(pi=[256,256],qf=[256, 256]))
                        ]  
    )
 

############################################################################################
############################################################################################
############################################################################################

if __name__ == "__main__":
    run(Args())