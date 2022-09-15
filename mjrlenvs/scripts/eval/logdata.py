import numpy as np
import pandas as pd 
import json 
import os
from scipy.interpolate import interp1d, make_interp_spline, BSpline 
 

class LogData():
    def __init__(self, file_path=None) -> None:
        if file_path is not None:
            with open(file_path, 'r') as f:
                self.data = json.loads(f.read())   

    def load(self, file_path): 
        with open(file_path, 'r') as f: 
            self.data = json.loads(f.read())  

    def num_episodes(self):  
        return len(self.data["episodes_return"])

    def num_steps(self):   
        return int(self.data["num_tot_steps"])
         
    def get_data(self, episode):  
        return self.data["episodes_data"][episode]  
     
    def get_obs(self, episode):
        episodes_data = self.get_data(episode) 
        obs = episodes_data["obs"]
        return obs

    def get_rewards(self, episode):
        episodes_data = self.get_data(episode) 
        rewards = episodes_data["rewards"]
        return rewards

    def get_actions(self, episode):
        episodes_data = self.get_data(episode) 
        actions = episodes_data["actions"]
        return actions

    def get_steps(self, episode):
        episodes_data = self.get_data(episode) 
        steps = episodes_data["steps"]
        return steps

    def get_episodes_return(self): 
        return self.data["episodes_return"] 
 
    def df_steps_rewards(self,episode):
        return pd.DataFrame(dict(
                    Steps = self.get_steps(episode), 
                    Rewards = self.get_rewards(episode)
                    )
                )

    def _smooth(self,timeframe,data): 
        newtimeframe = np.linspace(timeframe.min(), timeframe.max(), self.num_steps())
        # smooth_returns = interp1d(timeframe, returns, kind='cubic')  
        # returns = smooth_returns(newtimeframe)  
        spl = make_interp_spline(timeframe, data, k=3)  # type: BSpline
        data = spl(newtimeframe)
        timeframe = newtimeframe
        return timeframe, data 
 
    def df_episodes_returns(self, smooth=False):
        data = self.get_episodes_return()
        timeframe = np.arange(self.num_episodes())
        if smooth:
            timeframe, data = self._smooth(timeframe,data)
        return pd.DataFrame(dict(Episodes = timeframe, Returns = data))  
  
    def df_runs_episodes_returns(self, folder_path, smooth=False): 
        comb_df = pd.DataFrame()
        for file_name in os.listdir(folder_path):
            name,ext = os.path.splitext(file_name)
            if name.startswith("log_") and ext==".json": 
                file_path = os.path.join(folder_path,file_name)
                self.load(file_path)
                df = self.df_episodes_returns(smooth)
                df["Runs"] = [name]*len(df["Episodes"])
                comb_df = pd.concat([comb_df, df])
        return comb_df

