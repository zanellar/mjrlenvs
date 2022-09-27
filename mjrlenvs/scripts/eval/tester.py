import os 
import json
import random
from stable_baselines3 import HER, SAC, TD3, DDPG  
from stable_baselines3.common.callbacks import CallbackList, BaseCallback 
from mjrlenvs.scripts.args.pkgpaths import PkgPath
from mjrlenvs.scripts.env.envutils import wrapenv  
 
 

class TestRun():

    def __init__(self, run_args, render=None) -> None:

        self.args = run_args 
        
        self.env = self.args.ENV_EVAL 
        self.env = wrapenv(self.env, self.args, load_norm_env = True)  

        self.callback_list = CallbackList([])
        self.cb = None
 
 
        ###### INPUT FOLDERS PATH
        out_train_folder = PkgPath.OUT_TRAIN_FOLDER if self.args.OUT_TRAIN_FOLDER is None else self.args.OUT_TRAIN_FOLDER  
        self.training_output_folder_path = os.path.join(out_train_folder, self.args.ENVIRONMENT, self.args.RUN_ID)
        self.saved_training_logs_path = os.path.join(self.training_output_folder_path, "logs")  
        print(f"Loading logs from: {self.training_output_folder_path}")

        ###### OUTPUT FOLDERS PATH
        out_test_folder = PkgPath.OUT_TEST_FOLDER if self.args.OUT_TEST_FOLDER is None else self.args.OUT_TEST_FOLDER
        self.testing_output_folder_path = os.path.join(out_test_folder, self.args.ENVIRONMENT, self.args.RUN_ID)   
 
        if not os.path.exists(self.testing_output_folder_path):
            os.makedirs(self.testing_output_folder_path)


    def _loadmodel(self, model_id="first"):  
        if not model_id.startswith(("SAC","DDPG","TD3")): 
            list_train_name = os.listdir(self.saved_training_logs_path)
            if model_id=="first": # take the first train of this run
                index = 0
            elif model_id=="random":
                index = random.randint(0,len(list_train_name)-1)
            elif model_id=="last": 
                index = -1  
            train_file_name = list_train_name[index] 
            model_id = os.path.splitext(train_file_name)[0].split(sep="_", maxsplit=1)[1] 

        saved_model_path = os.path.join(self.training_output_folder_path, "checkpoints", model_id, "best_model", "best_model.zip")
  
        if self.args.AGENT == "SAC": 
            self.model = SAC.load(saved_model_path)
        if self.args.AGENT == "DDPG": 
            self.model = DDPG.load(saved_model_path)
        if self.args.AGENT == "TD3": 
            self.model = TD3.load(saved_model_path)
        
        return model_id

    def registercallback(self,cb): # TODO to be tested 
        self.cb = cb

    def eval_returns_model(self, model_id="random", n_eval_episodes=30, render=False, save=False):  # TODO call loadmodel inside the method
        self._loadmodel(model_id)
        obs = self.env.reset()  
        returns_list = []
        episode_return = 0
        i = 0
        while i<=n_eval_episodes: 
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)   
            episode_return += reward.item()
            if render:
                self.env.render() # BUG not working cam selection
            if done:
                i +=1 
                obs = self.env.reset()
                returns_list.append(episode_return) 
                episode_return = 0 
        
        if save:
            file_path =  os.path.join(self.testing_output_folder_path, f"{model_id}.txt") 
            with open(file_path, 'w') as file:  
                file.write(returns_list)  

        return returns_list 

    def infer(self, model_id="random", num_episodes=3, render = True, cam = "frontview"): # TODO call loadmodel inside the method
        self._loadmodel(model_id)
        obs = self.env.reset() 
        i = 0
        while i<=num_episodes: 
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            if render:
                self.env.render(cam) # BUG not working cam selection
            if done:
                i +=1 
                obs = self.env.reset()

    def eval_returns_run(self, n_eval_episodes=30, render=False, save=False, plot=False):  
        data = {}
        for file_name in os.listdir(self.saved_training_logs_path):  
            name = os.path.splitext(file_name)[0]
            prefix, model_id = name.split(sep="_", maxsplit=1)
            if prefix == "log":   
                print(f"Evaluating {model_id}")
                returns_list = self.eval_returns_model(model_id=model_id, n_eval_episodes=n_eval_episodes, render=render, save=False)
                data[model_id] = returns_list
        if plot:
            pass #TODO statannotation 
        if save:  
            file_path =  os.path.join(self.testing_output_folder_path, "returns_eval_run.json") 
            with open(file_path, 'w') as f:
                json.dump(data, f) 
        return data
 