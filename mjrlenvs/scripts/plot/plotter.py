import os 
import matplotlib.pyplot as plt
import seaborn as sns  
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

    def plot_avg_train_run_returns(self, env_name, run_id, labels=[], save=True, show=True ):  
 
        training_output_folder_path = os.path.join(self.out_train_folder, env_name, run_id)
        saved_training_logs_path = os.path.join(training_output_folder_path, "logs")    
    
        save_path = os.path.join(self.save_training_plots_folder_path, f"returns_train_{run_id}.pdf")
        data = df_run_episodes_returns(logs_folder_path=saved_training_logs_path, smooth=True)

        plt.figure() 
        sns.set(style="darkgrid", font_scale=1.5) 

        ax = sns.lineplot(
            data = data, 
            x =  "Episodes", 
            y = "Returns",   
            estimator = np.mean,
            errorbar='sd'
        )  

        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=len(labels), frameon=False, labels=labels)

        if save:
            plt.savefig(save_path, bbox_inches='tight', format="pdf") 
        if show: 
            plt.show()
  
    ##############################################################################################################

    def plot_avg_train_multirun_returns(self, env_run_ids, labels=[], save=True, show=True, plot_name=None): 
        if plot_name is None:
            plot_name = str(len(env_run_ids)) 
        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]  
        data = df_multiruns_episodes_returns(run_paths_list=run_paths_list, smooth=True ) 
        save_path = os.path.join(self.save_multirun_training_plots_path, f"returns_train_multirun_{plot_name}.pdf")

        plt.figure() 
        sns.set(style="darkgrid", font_scale=1.5) 

        ax = sns.lineplot(
            data = data, 
            x =  "Episodes", 
            y = "Returns",  
            hue = "Runs",
            estimator = np.mean,
            errorbar='sd'
        )  

        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=len(labels), frameon=False, labels=labels)

        if save:
            plt.savefig(save_path, bbox_inches='tight', format="pdf") 
        if show: 
            plt.show()
 