from cProfile import run
import os 
from mjrlenvs.scripts.plot.plotutils import *  
from mjrlenvs.scripts.plot.datautils import *
from mjrlenvs.scripts.args.pkgpaths import PkgPath

class Plotter():

    def __init__(self, out_train_folder=None,out_test_folder=None, plot_folder=None) -> None: 
 
        ###### INPUT FOLDERS PATH
        self.out_train_folder = PkgPath.OUT_TRAIN_FOLDER if out_train_folder is None else out_train_folder   
        self.out_test_folder = PkgPath.OUT_TEST_FOLDER if out_test_folder is None else out_test_folder
  
        ###### OUTPUT FOLDERS PATH 
        self.plots_folder_path = PkgPath.PLOT_FOLDER  if plot_folder is None else plot_folder

        self.save_testing_plots_folder_path = os.path.join(self.plots_folder_path,"testing")
        if not os.path.exists(self.save_testing_plots_folder_path):
            os.makedirs(self.save_testing_plots_folder_path)

        self.save_training_plots_folder_path = os.path.join(self.plots_folder_path,"training") 
        if not os.path.exists(self.save_training_plots_folder_path):
            os.makedirs(self.save_training_plots_folder_path)
 
        self.save_multirun_testing_plots_path = os.path.join(self.save_testing_plots_folder_path, "multirun" )
        if not os.path.exists(self.save_multirun_testing_plots_path):
            os.makedirs(self.save_multirun_testing_plots_path)

        self.save_multirun_training_plots_path = os.path.join(self.save_training_plots_folder_path, "multirun" )
        if not os.path.exists(self.save_multirun_training_plots_path):
            os.makedirs(self.save_multirun_training_plots_path)

    ##############################################################################################################

    def plot_avg_train_run_returns(self, env_name, run_id, save=True, show=True, save_path=None):  
 
        training_output_folder_path = os.path.join(self.out_train_folder, env_name, run_id)
        saved_training_logs_path = os.path.join(training_output_folder_path, "logs")    
   
        if save_path is None:
            save_path = os.path.join(self.save_training_plots_folder_path, f"returns_train_{run_id}.pdf")
 
        mean_std_plt(
            data = df_run_episodes_returns(logs_folder_path=saved_training_logs_path, smooth=True), 
            title = 'Average Train Episode Return',
            xaxis = "Episodes", 
            value = "Returns",  
            estimator = np.mean,
            save = save, 
            show = show, 
            save_path = save_path
        )
  
    ##############################################################################################################

    def plot_avg_train_multirun_returns(self, env_run_ids, save=True, show=True, plot_name=None): 
        if plot_name is None:
            plot_name = str(len(env_run_ids)) 
        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]  
        mean_std_plt(
            data = df_multiruns_episodes_returns(run_paths_list=run_paths_list, smooth=True ), 
            title = 'Average Train Episode Return Multirun',
            xaxis = "Episodes", 
            value = "Returns",  
            hue = "Runs",
            estimator = np.mean,
            save = save, 
            show = show, 
            save_path = os.path.join(self.save_multirun_training_plots_path, f"returns_train_multirun_{plot_name}.pdf"), 
        ) 