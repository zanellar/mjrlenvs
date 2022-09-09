 
from ast import Try
from pickle import FALSE
import numpy as np 
from stable_baselines3 import HER, SAC, TD3, DDPG 
from stable_baselines3.common.evaluation import evaluate_policy   
from stable_baselines3.common.callbacks import CallbackList, BaseCallback 
from mjrlenvs.scripts.pkgpaths import PkgPath
from mjrlenvs.scripts.envutils import wrapenv 
from mjrlenvs.scripts.plotdata import PlotLogData  
 
 

class TestAgent():

    def __init__(self, run_args, rendering=None) -> None:

        self.args = run_args 
        
        self.env = self.args.ENV_EVAL 
        self.env = wrapenv(self.env, self.args, load_norm_env = True)  

        self.callback_list = CallbackList([])
        self.cb = None

    def loadmodel(self, name=None): 
        saved_model_path = PkgPath.trainingdata(f"{self.args.ENVIRONMENT}/{self.args.RUN_ID}/checkpoints/{name}/best_model/best_model.zip")

        if self.args.AGENT == "SAC": 
            self.model = SAC.load(saved_model_path)
        if self.args.AGENT == "DDPG": 
            self.model = DDPG.load(saved_model_path)
        if self.args.AGENT == "TD3": 
            self.model = TD3.load(saved_model_path)

    def registercallback(self,cb):
        # self.callback_list.callbacks.append(cb)
        self.cb = cb

    def evalpolicy(self, n_eval_episodes=30, render=False): 
        mean_reward, std_reward = evaluate_policy(
                                    self.model,
                                    self.env, 
                                    n_eval_episodes = n_eval_episodes,  
                                    render = render, 
                                    deterministic = True,
                                    callback = self.cb
                                    )
 
        # print(mean_reward, std_reward)
        return mean_reward, std_reward

    def infer(self, rendering = True, cam = "frontview"):
        obs = self.env.reset() 
        while True: 
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            if rendering:
                self.env.render(cam)
            if done:
                obs = self.env.reset()
    
    def plot(self, name=None, y="returns"): 
        save_training_logs_file_path = PkgPath.trainingdata(f"{self.args.ENVIRONMENT}/{self.args.RUN_ID}/logs")
        file_name = f"log_{name}.json"
        plotter = PlotLogData(save_training_logs_file_path, file_name=file_name)
        if y == "returns":
            plotter.plt_returns()
        else:
            plotter.plt_rewards(episode=y) 




 