
import os
from mjrlenvs.scripts.plot.datautils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd   
import numpy as np
import seaborn as sns
 
def mean_std_plt(data, title='Title', xaxis="timeframe", value="data", hue=None, save=True, show=True, save_path="plot.pdf", **kwargs):
    '''https://www.reddit.com/r/reinforcementlearning/comments/gnvlcp/way_to_plot_goodlooking_rewards_plots/'''

    print(f"Plotting: {title}")
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

def stat_boxes_plt(data, title='Title', xaxis="set", value="data", save=True, show=True, save_path="plot.pdf", **kwargs ):
    print(f"Plotting: {title}")
    plt.figure() 

    sns.set(style="darkgrid", font_scale=1.5) 
    # sns.boxplot(data=data, x=xaxis, y=value, orient="v", **kwargs)
    sns.histplot(data=data, x=xaxis, y=value, kde=True, **kwargs)
    
    plt.legend(loc='best').set_draggable(True) 

    if save:
        plt.savefig(save_path, bbox_inches='tight', format="pdf") 
    if show: 
        plt.show()
 