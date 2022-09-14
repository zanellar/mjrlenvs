
import os
from mjrlenvs.scripts.logdata import LogData
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd   
import numpy as np
import seaborn as sns


class PlotLogData():
 

    def __init__(self, logs_folder_path, file_name=None) -> None:  
        self.logs_folder_path =  logs_folder_path 
        self.data = LogData()    
        if file_name is not None: 
            self.data.load(file_path=os.path.join(logs_folder_path,file_name))
    
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
    
    def plt_returns(self, save=True, show=True, save_name="plot.pdf", smooth=True):  
        
        df_logs = self.data.df_runs_episodes_returns(self.logs_folder_path, smooth)

        plt.figure()
        PlotLogData._stat_plot(
            df_logs, 
            title='Average Episode Return',
            xaxis="Episodes", 
            value="Returns",  
            estimator=np.mean
        )

        if save:
            plt.savefig(save_name, bbox_inches='tight', format="pdf") 
        if show: 
            plt.show()  

    def plt_rewards(self, episode): 
        rewards = self.data.df_steps_rewards(episode)
        rewards.plot(x="Steps",y='Rewards', grid=True, legend=False)
        plt.xlabel('steps')
        plt.ylabel('rewards') 
        plt.show()
