 
import time 
from stable_baselines3.common.callbacks import BaseCallback

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