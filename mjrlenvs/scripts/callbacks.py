
import os 
import time
import pandas as pd
import json
from stable_baselines3.common.callbacks import BaseCallback

############################################################################################

class SaveTrainingLogsCallback(BaseCallback):
    def __init__(self, folder_path, file_name, no_rollouts_episode=1, save_all=False):
        super().__init__() 
        self.save_all = save_all
        self.folder_path = folder_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.file_path = os.path.join(self.folder_path,f"log_{file_name}.json")
        self.no_rollouts_episode = no_rollouts_episode 
        self.episodes = 0
        self.episodes_return = 0
        self.rollouts_return = 0 

    def _on_training_start(self) -> None:
        self.episodes_data = dict(steps=[], obs=[], actions=[], rewards=[])
        self.training_data = dict(episodes_data=[], episodes_return=[], rollouts_return=[])
        self.rollouts = 0   
        return True

    def _on_rollout_start(self) -> None:
        self.rollouts_return = 0 
        if self.rollouts % self.no_rollouts_episode == 0:
            self.episodes_return = 0   
        return True

    def _on_step(self) -> bool:
        sample = self.training_env.env_method("get_sample") 
        obs, action, reward = sample[0]
        if self.save_all:
            self.episodes_data["steps"].append(self.num_timesteps)
            self.episodes_data["obs"].append(obs.tolist())
            self.episodes_data["actions"].append(action.tolist())
            self.episodes_data["rewards"].append(reward) 
        self.rollouts_return += reward
        return True
    
    def _on_rollout_end(self) -> None: 
        # df_ep = pd.DataFrame(self.episodes_data)
        self.rollouts += 1
        self.episodes_return += self.rollouts_return
        self.training_data["rollouts_return"].append(self.rollouts_return)  
        if self.rollouts % self.no_rollouts_episode == 0:   
            self.training_data["episodes_return"].append(self.episodes_return) 
            if self.save_all:
                self.training_data["episodes_data"].append(self.episodes_data)  
            self.episodes += 1 
        return True

    def _on_training_end(self) -> bool: 
        print("@@@@@@@@@@@ TRAINING END @@@@@@@@@@@@@@") 
        self.training_data["num_tot_steps"]= self.num_timesteps
        with open(self.file_path, 'w') as f:
            json.dump(self.training_data, f)

        # df = pd.DataFrame(self.training_data)
        # df.to_csv(self.file_path,sep="\t")
         
############################################################################################

class Time2EndCallback(BaseCallback):
    def __init__(self, repete_times, number_configs,number_episodes,episode_horizon,config_index,training_index ):
        super().__init__( )
        self.t0 = time.time() 
        self.t1 = self.t0  
        self.repete_times = repete_times
        self.number_configs = number_configs
        self.number_episodes = number_episodes,
        self.episode_horizon = episode_horizon,
        self.config_index = config_index,
        self.training_index = training_index

    def _on_step(self) -> bool:
        self.t2 = time.time()  
        min2end =  (self.t2-self.t1)*(self.episode_horizon[0]*self.number_episodes[0] - self.num_timesteps)/60 
        self.logger.record('time2end/training[minutes]', min2end)  # -> tensorboard 
        self.t1 = self.t2 
        return True

    def _on_training_end(self) -> bool: 
        ti = time.time()  
        hours2end = ((ti-self.t0)*(self.number_configs*self.repete_times - self.config_index[0]*self.repete_times - self.training_index)/60)/60
        self.logger.record('time2end/total[hours]', hours2end)  # -> tensorboard 
