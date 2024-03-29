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

def _interpolate(timeframe, values, timesteps): 
    newtimeframe = np.linspace(timeframe.min(), timeframe.max(), int(timesteps )) 
    spl = make_interp_spline(timeframe, values, k=3)  # type: BSpline
    values = spl(newtimeframe)
    timeframe = newtimeframe
    return timeframe, values 

def _smooth(values, weight=0.5):
    smoothed = np.array(values)
    for i in range(1, smoothed.shape[0]):
        smoothed[i] = smoothed[i-1] * weight + (1 - weight) * smoothed[i]
    return smoothed
  
def df_episodes_returns(data, smooth=False, interpolate=False):
    '''
    DataFrame with the returns corresponding to a specific episode  
    @param data: data from the log file (of an episode)
    @param smooth: whether to smooth the returns
    @param interpolate: whether to interpolate the returns
    '''
    returns = get_episodes_return(data)
    timeframe = np.arange(num_episodes(data))
    if interpolate:
        timeframe, returns = _interpolate(timeframe,returns,num_steps(data))
    if smooth:
        returns = _smooth(returns)
    return pd.DataFrame(dict(Episodes = timeframe, Returns = returns))  

def df_run_episodes_returns(run_folder_path, smooth=False, interpolate=False): 
    '''
    DataFrame with the returns corresponding to each episode of the given run
    @param run_folder_path: path to the run folder
    @param smooth: whether to smooth the returns
    @param interpolate: whether to interpolate the returns
    '''
    comb_df = pd.DataFrame() 
    print(f"Loading logs from: {run_folder_path}")
    saved_logs_path = os.path.join(run_folder_path, "logs") 
    for file_name in os.listdir(saved_logs_path):
        name,ext = os.path.splitext(file_name)
        if name.startswith("log_") and ext==".json": 
            file_path = os.path.join(saved_logs_path,file_name)
            data = dataload(file_path)
            df = df_episodes_returns(data,smooth=smooth, interpolate=interpolate)
            df["Trainings"] = [name]*len(df["Episodes"])
            comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df

def df_multiruns_episodes_returns( run_paths_list, smooth=False, interpolate=False, run_label_list=[] ):  
    '''
    DataFrame with the returns corresponding to each run of the given list of runs
    @param run_paths_list: list of paths to the run folders
    @param smooth: whether to smooth the returns
    @param interpolate: whether to interpolate the returns
    @param run_label_list: list of labels for the runs
    '''
    comb_df = pd.DataFrame()   
    for i,run_folder_path in enumerate(run_paths_list): 
        df = df_run_episodes_returns(run_folder_path, smooth=smooth, interpolate=interpolate ) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else: 
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id   
        df["Runs"] = [run_label]*len(df["Trainings"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 
    
def multirun_steps(run_paths_list):
    num_steps_list = []
    for i, run_folder_path in enumerate(run_paths_list): 
        saved_logs_path = os.path.join(run_folder_path, "logs") 
        run_num_steps_list = []
        for file_name in os.listdir(saved_logs_path):  
            name, ext = os.path.splitext(file_name)   
            if name.startswith("log_") and ext==".json": 
                file_path = os.path.join(saved_logs_path, file_name)
                data = dataload(file_path) 
                run_num_steps_list += [num_steps(data)]
        num_steps_list += [run_num_steps_list]
    return num_steps_list


###########################################################################
###########################################################################
###########################################################################

def df_test_run_returns(run_folder_path, smooth=False): 
    ''' 
    DataFrame with the returns corresponding to each episode of all the tests in the given run
    @param run_folder_path: path to the run folder
    @param smooth: whether to smooth the returns
    ''' 
    print(f"Loading logs from: {run_folder_path}")
    saved_returns_test_path = os.path.join(run_folder_path, "returns_eval.json")  
    data = dataload(saved_returns_test_path)
    returns_values = [] 
    for v in data.values(): 
        val = np.array(v)
        returns_values = np.concatenate([returns_values,val]) 
    df = pd.DataFrame(dict(Tests = np.arange(len(returns_values)), Returns = returns_values))    
    return df

def df_test_multirun_returns(run_paths_list, smooth=False, run_label_list=[]):
    '''
    DataFrame with the returns corresponding to each test of the given list of runs
    @param run_paths_list: list of paths to the run folders
    @param smooth: whether to smooth the returns
    @param run_label_list: list of labels for the runs
    '''
    comb_df = pd.DataFrame()   
    for i,run_folder_path in enumerate(run_paths_list): 
        df = df_test_run_returns(run_folder_path, smooth) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else: 
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id  
        df["Runs"] = [run_label]*len(df["Tests"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 

 