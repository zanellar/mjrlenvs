import os 
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

        ###### INPUT FOLDERS PATH
        if self.args.RUN_OUT_FOLDER_PATH is not None:
            self.training_output_folder_path = self.args.RUN_OUT_FOLDER_PATH
        else:
            self.training_output_folder_path = os.path.join(PkgPath.OUT_TRAIN_FOLDER,f"{self.args.ENVIRONMENT}/{self.args.RUN_ID}")

        print(f"Loading logs from: {self.training_output_folder_path}")
        self.saved_training_logs_path = os.path.join(self.training_output_folder_path,"logs") 

        ###### OUTPUT FOLDERS PATH
        self.testing_output_folder_path = os.path.join(PkgPath.OUT_TEST_FOLDER,f"{self.args.ENVIRONMENT}/{self.args.RUN_ID}") 

        self.save_testing_evals_path = os.path.join(self.testing_output_folder_path,"evals")
        if not os.path.exists(self.save_testing_evals_path):
            os.makedirs(self.save_testing_evals_path)

        self.save_testing_plots_path = os.path.join(self.testing_output_folder_path,"plots")
        if not os.path.exists(self.save_testing_plots_path):
            os.makedirs(self.save_testing_plots_path)

    def loadmodel(self, name): 
        saved_model_path = os.path.join(self.training_output_folder_path,f"/checkpoints/{name}/best_model/best_model.zip")

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
    
    def plot(self, model_id=None, y="returns",save=True, show=True, save_name=None): 
        if model_id is not None:
            file_name = f"log_{model_id}.json"
        else:
            file_name = None    
        plotter = PlotLogData(self.saved_training_logs_path, file_name=file_name)   
        if y == "returns":
            if save_name is None:
                save_name = os.path.join(self.save_testing_plots_path,f"{self.args.RUN_ID}.pdf")
            plotter.plt_returns(save=save, show=show, save_name=save_name)
        else:
            plotter.plt_rewards(episode=y) 
 



 