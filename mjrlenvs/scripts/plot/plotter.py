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

    def _line_plot(self,data,x,y,hue,xsteps,run_paths_list,labels,xlim,ylim,xlabels,ylabels,show,save,save_path,ext): 

        plt.figure() 
        sns.set(style="darkgrid", font_scale=1.5) 

        ax = sns.lineplot(
            data = data, 
            x = x, 
            y = y,  
            hue = hue
        )  

        if xsteps:
            nstps = multirun_steps(run_paths_list)[0][0]
            ax.set_xticklabels(range(0,nstps))
            ax.set_xlabel('Steps')  
        
        if xlabels:
            ax.set_xlabel(xlabels)  
        if ylabels: 
            ax.set_ylabel(ylabels)  

        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=len(labels), frameon=False )
        if xlim[0] is not None and xlim[1] is not None:
            plt.xlim(xlim[0],xlim[1])
        if ylim[0] is not None and ylim[1] is not None:
            plt.ylim(ylim[0],ylim[1])


        if save: 
            plt.savefig(save_path, bbox_inches='tight', format=ext) 
        if show: 
            plt.show()
  
    ##############################################################################################################
 
    def _stat_plot(self,data,x,y,hue,plot_type,labels,xlabels,ylabels,show,save,save_path,ext):

        plt.figure() 
        sns.set(style="darkgrid", font_scale=1.5)  

        if plot_type == "histplot": 
            ax = sns.histplot(
                data = data,  
                x = y,  
                hue = x,
                kde = True,
                bins = 10 
            )   
            labels.reverse()
            ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=len(labels), frameon=False, labels=labels)

        if plot_type == "boxplot": 
            ax = sns.boxplot(
                data = data,  
                x = x, 
                y = y,   
                hue = hue,
                orient = "v"
            )
            ax.set_xticklabels(labels)

        if plot_type == "violinplot": 
            ax = sns.violinplot(
                data = data, 
                x = x, 
                y = y,   
                hue = hue,
                orient = "v",
                bw = 0.2,  cut=0
            )
            ax.set_xticklabels(labels)
  
        if xlabels:
            ax.set_xlabel(xlabels)  
        if ylabels: 
            ax.set_ylabel(ylabels)  

 
        if save: 
            plt.savefig(save_path, bbox_inches='tight', format=ext) 
        if show: 
            plt.tight_layout() 
            plt.show()
        plt.close()

    ##############################################################################################################
 
    def multirun_returns_train(self, env_run_ids, labels=[], xsteps=False, smooth=False, save=True, show=True, plot_name=None, ext="pdf", xlim=[None,None], ylim=[None,None],xlabels=None,ylabels=None):
        if plot_name is None:
            plot_name = str(len(env_run_ids)) 

        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]  
        data = df_multiruns_episodes_returns(run_paths_list=run_paths_list, smooth=smooth, interpolate=xsteps, run_label_list=labels ) 
        save_path = os.path.join(self.save_multirun_training_plots_path, f"{plot_name}_multirun_returns_train.{ext}") 

        self._line_plot(
            data = data,
            x =  "Episodes", 
            y = "Returns",  
            hue = "Runs",
            xsteps = xsteps,
            run_paths_list = run_paths_list,
            labels = labels,
            xlim = xlim,
            ylim = ylim,
            xlabels = xlabels,
            ylabels = ylabels,
            show = show,
            save = save,
            save_path = save_path,
            ext = ext
        )


    ##############################################################################################################################################################
 
    def multirun_returns_test(self, env_run_ids, labels=[], xlabels=None, ylabels=None,  save=True, show=True, plot_name=None, plot_type="histplot", ext="pdf"): 
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_test_folder, env_run) for env_run in env_run_ids]
        
        data = df_test_multirun_returns(run_paths_list=run_paths_list) 
        save_path = os.path.join(self.save_multirun_testing_plots_path, f"{plot_name}_{plot_type}_multirun_returns_test.{ext}")

        self._stat_plot(
            data = data,
            x =  "Runs", 
            y = "Returns",  
            hue = None,  
            plot_type = plot_type,
            labels = labels, 
            xlabels = xlabels,
            ylabels = ylabels,
            show = show,
            save = save,
            save_path = save_path,
            ext = ext
        )    

 