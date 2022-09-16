
import os
from mjrlenvs.scripts.eval.logdata import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd   
import numpy as np
import seaborn as sns
 
def mean_std_plt(data, title='Title', xaxis="timeframe", value="data", hue=None, save=True, show=True, save_path="plot.pdf", **kwargs):
    '''https://www.reddit.com/r/reinforcementlearning/comments/gnvlcp/way_to_plot_goodlooking_rewards_plots/'''

    plt.figure() 
    sns.set(style="darkgrid", font_scale=1.5) 
    sns.lineplot(data=data, x=xaxis, y=value, hue=hue, errorbar='sd', **kwargs)  
    plt.legend(loc='best').set_draggable(True)  
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale: 
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) 
    plt.tight_layout(pad=0.5)


    if save:
        plt.savefig(save_path, bbox_inches='tight', format="pdf") 
    if show: 
        plt.show()

def stat_boxes_plt(data, title='Title', xaxis="timeframe", value="data", **kwargs ):
    sns.set(style="darkgrid", font_scale=1.5) 
    sns.boxplot(data=data, x=xaxis, y=value, **kwargs)
    plt.legend(loc='best').set_draggable(True) 

##################################################################################################################

def plt_run_returns(logs_folder_path, smooth=True, show=True, save=True, save_path="plot.pdf"):   
    print("Plotting Average Episode Return") 
    mean_std_plt(
        data = df_run_episodes_returns(logs_folder_path, smooth), 
        title = 'Average Episode Return',
        xaxis = "Episodes", 
        value = "Returns",  
        estimator = np.mean,
        save = save, 
        show = show, 
        save_path = save_path
    )
    
def plt_episode_rewards(logs_folder_path, file_name, episode):  
    data = dataload(file_path = os.path.join(logs_folder_path,file_name))
    rewards = df_steps_rewards(data, episode)
    rewards.plot(x="Steps",y='Rewards', grid=True, legend=False)
    plt.xlabel('steps')
    plt.ylabel('rewards') 
    plt.show()

def plt_multirun_return(env_run_paths, show=True, save=True, save_path="plot.pdf", smooth=True):  
    mean_std_plt(
        data = df_multiruns_episodes_returns(env_run_paths, smooth ), 
        title = 'Average Episode Return Multirun',
        xaxis = "Episodes", 
        value = "Return",  
        hue = "Runs",
        estimator = np.mean,
        save = save, 
        show = show, 
        save_path = save_path
    )