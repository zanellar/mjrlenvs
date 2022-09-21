import numpy as np
import pandas as pd 
import json 
import os
from scipy.interpolate import interp1d, make_interp_spline, BSpline 
 
 
def dataload(file_path): 
    with open(file_path, 'r') as f: 
        data = json.loads(f.read())  
    return data

def num_episodes(data):  
    return len(data["episodes_return"])

def num_steps(data):   
    return int(data["num_tot_steps"])
        
def get_data(data, episode):  
    return data["episodes_data"][episode]  
    
def get_obs(episode):
    episodes_data = get_data(episode) 
    obs = episodes_data["obs"]
    return obs

def get_rewards(episode):
    episodes_data = get_data(episode) 
    rewards = episodes_data["rewards"]
    return rewards

def get_actions(episode):
    episodes_data = get_data(episode) 
    actions = episodes_data["actions"]
    return actions

def get_steps(episode):
    episodes_data = get_data(episode) 
    steps = episodes_data["steps"]
    return steps

def get_episodes_return(data): 
    return data["episodes_return"] 

def df_steps_rewards(data,episode):
    return pd.DataFrame(dict(
                Steps = get_steps(data, episode), 
                Rewards = get_rewards(data, episode)
                )
            )

def _smooth(timeframe, values, timesteps): 
    newtimeframe = np.linspace(timeframe.min(), timeframe.max(), timesteps)  
    spl = make_interp_spline(timeframe, values, k=3)  # type: BSpline
    values = spl(newtimeframe)
    timeframe = newtimeframe
    return timeframe, values 

def df_episodes_returns(data, smooth=False):
    returns = get_episodes_return(data)
    timeframe = np.arange(num_episodes(data))
    if smooth:
        timeframe, returns = _smooth(timeframe,returns,num_steps(data))
    return pd.DataFrame(dict(Episodes = timeframe, Returns = returns))  

def df_run_episodes_returns(run_folder_path, smooth=False): 
    comb_df = pd.DataFrame() 
    print(f"Loading logs from: {run_folder_path}")
    saved_logs_path = os.path.join(run_folder_path, "logs") 
    for file_name in os.listdir(saved_logs_path):
        name,ext = os.path.splitext(file_name)
        if name.startswith("log_") and ext==".json": 
            file_path = os.path.join(saved_logs_path,file_name)
            data = dataload(file_path)
            df = df_episodes_returns(data,smooth)
            df["Trainings"] = [name]*len(df["Episodes"])
            comb_df = pd.concat([comb_df, df])
    return comb_df

def df_multiruns_episodes_returns( run_paths_list, smooth=False ):  
    comb_df = pd.DataFrame()   
    for run_folder_path in run_paths_list: 
        df = df_run_episodes_returns(run_folder_path, smooth) 
        env_id, run_id  = run_folder_path.split("/")[-2:]  
        df["Runs"] = [env_id+"_"+run_id]*len(df["Trainings"])
        comb_df = pd.concat([comb_df, df])
    return comb_df 
    