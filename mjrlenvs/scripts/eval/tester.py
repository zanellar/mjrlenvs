import os 
from ast import Try
from pickle import FALSE
import numpy as np 
from stable_baselines3 import HER, SAC, TD3, DDPG 
from stable_baselines3.common.evaluation import evaluate_policy   
from stable_baselines3.common.callbacks import CallbackList, BaseCallback 
from mjrlenvs.scripts.args.pkgpaths import PkgPath
from mjrlenvs.scripts.env.envutils import wrapenv 
from mjrlenvs.scripts.eval.plotdata import *  
 
 

class TestRun():

    def __init__(self, run_args, rendering=None) -> None:

        self.args = run_args 
        
        self.env = self.args.ENV_EVAL 
        self.env = wrapenv(self.env, self.args, load_norm_env = True)  

        self.callback_list = CallbackList([])
        self.cb = None
 
        ###### INPUT FOLDERS PATH
        out_train_folder = PkgPath.OUT_TRAIN_FOLDER if self.args.OUT_TRAIN_FOLDER is None else self.args.OUT_TRAIN_FOLDER 
        self.training_output_folder_path = os.path.join(out_train_folder, self.args.ENVIRONMENT, self.args.RUN_ID)

        print(f"Loading logs from: {self.training_output_folder_path}")
        self.saved_training_logs_path = os.path.join(self.training_output_folder_path, "logs") 

        ###### OUTPUT FOLDERS PATH
        out_test_folder = PkgPath.OUT_TEST_FOLDER if self.args.OUT_TEST_FOLDER is None else self.args.OUT_TEST_FOLDER
        self.testing_output_folder_path = os.path.join(out_test_folder, self.args.ENVIRONMENT, self.args.RUN_ID) 

        self.save_testing_evals_path = os.path.join(self.testing_output_folder_path, "evals")
        if not os.path.exists(self.save_testing_evals_path):
            os.makedirs(self.save_testing_evals_path)

        self.save_testing_plots_path = os.path.join(self.testing_output_folder_path, "plots")
        if not os.path.exists(self.save_testing_plots_path):
            os.makedirs(self.save_testing_plots_path)

    def loadmodel(self, name="first_trained"):  
        if name=="first_trained": # take the first train of this run
            train_file_name = os.listdir(self.saved_training_logs_path)[0] 
            name = os.path.splitext(train_file_name)[0].split(sep="_", maxsplit=1)[1] 

        saved_model_path = os.path.join(self.training_output_folder_path, "checkpoints", name, "best_model", "best_model.zip")
  
        if self.args.AGENT == "SAC": 
            self.model = SAC.load(saved_model_path)
        if self.args.AGENT == "DDPG": 
            self.model = DDPG.load(saved_model_path)
        if self.args.AGENT == "TD3": 
            self.model = TD3.load(saved_model_path)

    def registercallback(self,cb): # TODO to be tested
        # self.callback_list.callbacks.append(cb)
        self.cb = cb

    def evalpolicy(self, n_eval_episodes=30, render=False, save=False):  # TODO call loadmodel inside the method
        mean_reward, std_reward = evaluate_policy(
                                    self.model,
                                    self.env, 
                                    n_eval_episodes = n_eval_episodes,  
                                    render = render, 
                                    deterministic = True,
                                    callback = self.cb
                                    )
 
        print(f"mean_reward={mean_reward}, std_reward={std_reward}")
        return mean_reward, std_reward

    def infer(self, num_episodes=3, rendering = True, cam = "frontview"): # TODO call loadmodel inside the method
        obs = self.env.reset() 
        i = 0
        while i<=num_episodes: 
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            if rendering:
                self.env.render(cam) # BUG not working cam selection
            if done:
                i +=1 
                obs = self.env.reset()

    def evalrun(self, n_eval_episodes=30, render=False, save=False, plot=False):  
        lmean = []
        lstd = []
        for file_name in os.listdir(self.saved_training_logs_path):  
            name = os.path.splitext(file_name)[0]
            prefix, model_name = name.split(sep="_", maxsplit=1)
            if prefix == "log":  
                self.loadmodel(model_name)
                mean_train_return, std_train_return = self.evalpolicy(n_eval_episodes, render, save)
                lmean.append(mean_train_return)
                lstd.append(std_train_return) 
        if plot:
            pass #TODO statannotation 
        return lmean, lstd
 
    
    def plot(self, model_id=None, y="returns",save=True, show=True, save_path=None): 
        if model_id is not None:
            file_name = f"log_{model_id}.json"
        else:
            file_name = None     
        if y == "returns":
            if save_path is None:
                save_path = os.path.join(self.save_testing_plots_path, f"{self.args.RUN_ID}.pdf")
            plt_run_returns(
                logs_folder_path = self.saved_training_logs_path, 
                show=show, 
                save=save, 
                save_path=save_path
            )
        else:
            plt_episode_rewards(
                logs_folder_path = self.saved_training_logs_path, 
                file_name = file_name, 
                episode = y
            ) 
 

 
class TestMultiRun():

    def __init__(self, out_train_folder = None, out_test_folder = None) -> None:
        self.out_train_folder = PkgPath.OUT_TRAIN_FOLDER if out_train_folder is None else out_train_folder  
        self.out_test_folder = PkgPath.OUT_TEST_FOLDER if out_test_folder is None else out_test_folder  
        self.save_testing_plots_path = os.path.join(self.out_test_folder, "multirun", "plots")
        if not os.path.exists(self.save_testing_plots_path):
            os.makedirs(self.save_testing_plots_path)

    def evalmultirun(self):
        pass

    def plot(self, env_run_ids, save=True, show=True, plot_name=""):
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        env_run_paths = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]
        plt_multirun_return(
            env_run_paths, 
            show  = show, 
            save = save, 
            save_path = os.path.join(self.save_testing_plots_path, f"multirun_{plot_name}.pdf"), 
            smooth=True
        )