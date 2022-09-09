
import os
from mjrlenvs.scripts.logdata import LogData
import matplotlib.pyplot as plt
import pandas as pd 

class PlotLogData():
 

    def __init__(self, logs_folder_path, file_name=None) -> None:
        path = os.path.join(logs_folder_path,file_name)
        self.data = LogData(path)     

    def plt_returns(self, smooth=True): 
        returns = self.data.df_episode_return(smooth) 
        returns.plot(x="timeframe",y='data', grid=True, legend=False) 
        plt.xlabel("episodes" )
        plt.ylabel("returns") 
        plt.show()

    def plt_rewards(self, episode): 
        rewards = self.data.df_steps_rewards(episode)
        rewards.plot(x="timeframe",y='data', grid=True, legend=False)
        plt.xlabel('steps')
        plt.ylabel('rewards') 
        plt.show()

 