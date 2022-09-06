import pandas as pd 
import ast

from mjrlenvs.scripts.pkgpaths import PkgPath  

class LogData():
    def __init__(self, file_path) -> None:
        self.df = pd.read_csv(file_path, sep="\t")   
  
    def get_data(self, episode): 
        episode_data = self.df["episode_data"].loc[episode]
        episode_data = ast.literal_eval(episode_data)
        episode_data = pd.DataFrame(episode_data) 
        return episode_data 
     
    def no_episodes(self):  
        return len(self.df["episode_data"])
        
    def get_obs(self, episode):
        episode_data = self.get_data(episode) 
        obs = episode_data["obs"]
        return obs

    def get_rewards(self, episode):
        episode_data = self.get_data(episode) 
        rewards = episode_data["rewards"]
        return rewards

    def get_actions(self, episode):
        episode_data = self.get_data(episode) 
        actions = episode_data["actions"]
        return actions

    def get_steps(self, episode):
        episode_data = self.get_data(episode) 
        steps = episode_data["steps"]
        return steps

 
if __name__ == "__main__":
    logdata = LogData("/home/riccardo/projects/mjrlenvs/data/train/pendulum/prova2/logs/log_SAC_1_0.csv")
    
    print(logdata.no_episodes())