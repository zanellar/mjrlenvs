
import os
from mjrlenvs.scripts.logdata import LogData
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd   
import numpy as np
import seaborn as sns


class PlotLogData():
 

    def __init__(self, logs_folder_path, file_name=None) -> None:
        path = os.path.join(logs_folder_path,file_name)
        self.data = LogData(path)     
    
    @staticmethod
    def _stat_plot(data, title='Title', xaxis="timeframe", value="data", condition=None, **kwargs):
        '''https://www.reddit.com/r/reinforcementlearning/comments/gnvlcp/way_to_plot_goodlooking_rewards_plots/'''
 
        sns.set(style="darkgrid", font_scale=1.5)
 
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, errorbar='sd', **kwargs) 

        plt.legend(loc='best').set_draggable(True)

        xscale = np.max(np.asarray(data[xaxis])) > 5e3
        if xscale: 
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) 
        plt.tight_layout(pad=0.5)
    
    def plt_returns(self, smooth=True): 
        data = self.data.df_episode_return(smooth)  
        plt.figure()
        PlotLogData._stat_plot(
            data, 
            # xaxis="Episode", 
            # value="Average Episode Return", 
            # condition=condition, 
            # smooth=smooth, 
            estimator=np.mean
        )
        plt.show() 
        # returns.plot(x="timeframe",y='data', grid=True, legend=False) 
        # plt.xlabel("episodes" )
        # plt.ylabel("returns") 
        # plt.show()

    def plt_rewards(self, episode): 
        rewards = self.data.df_steps_rewards(episode)
        rewards.plot(x="timeframe",y='data', grid=True, legend=False)
        plt.xlabel('steps')
        plt.ylabel('rewards') 
        plt.show()
