import os 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
import seaborn as sns  
from mjrlenvs.scripts.plot.datautils import *
from mjrlenvs.scripts.args.pkgpaths import PkgPath

class Plotter():
    '''
    This class contains the methods to plot the results of the training and testing of the model.
    '''

    def __init__(self, out_train_folder=None,out_test_folder=None, plot_folder=None) -> None: 
        '''
        This method initializes the Plotter class by setting the paths of the folders where the results of the training and testing are saved.
        @param out_train_folder: path of the folder where the results of the training are saved.
        @param out_test_folder: path of the folder where the results of the testing are saved.
        @param plot_folder: path of the folder where the plots are saved.
        '''
 
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
    
    def _line_plot(self,data,x,y,hue,xsteps,run_paths_list,labels,xlim,ylim,save_path,xlabels=None,ylabels=None,show=False,save=True,ext="pdf",fontsize=25): 

        plt.figure() 
        sns.set(style="darkgrid", font_scale=1.5) 

        ax = sns.lineplot(
            data = data, 
            x = x, 
            y = y,  
            hue = hue,
            # errorbar=('ci', 99.7)
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
        plt.setp(ax.get_legend().get_texts(), fontsize=str(fontsize)) # for legend text

        if xlim[0] is not None and xlim[1] is not None:
            plt.xlim(xlim[0],xlim[1])
        if ylim[0] is not None and ylim[1] is not None:
            plt.ylim(ylim[0],ylim[1])


        if save: 
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path)) 
            plt.savefig(save_path, bbox_inches='tight', format=ext) 
        if show: 
            plt.show()
  
    ##############################################################################################################
    
    def _line_plot_nobs(self,maxvals,minvals,avgvals,labels,xlim,ylim,xlabels,ylabels,show,save,save_path,ext): 

        plt.figure() 
        sns.set(style="darkgrid", font_scale=1.5) 

        for label in avgvals.keys():
            plt.plot(range(len(avgvals[label])),avgvals[label], label = label)  
            plt.fill_between(range(len(avgvals[label])),  maxvals[label], minvals[label], alpha = .25)
  
        if xlabels:
            plt.xlabel(xlabels)  
        if ylabels: 
            plt.xlabel(ylabels)  

        # plt.legend()
        # plt.xlabel(xlabels)

        plt.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=len(labels), frameon=False )
        if xlim[0] is not None and xlim[1] is not None:
            plt.xlim(xlim[0],xlim[1])
        if ylim[0] is not None and ylim[1] is not None:
            plt.ylim(ylim[0],ylim[1])


        if save: 
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path)) 
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
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path)) 
            plt.savefig(save_path, bbox_inches='tight', format=ext) 
        if show: 
            plt.tight_layout() 
            plt.show()
        plt.close()

    ##############################################################################################################
 
    def multirun_returns_train(self, env_run_ids, labels=[], xsteps=False, smooth=False, save=True, show=True, plot_name=None, sub_folder_name="", ext="pdf", xlim=[None,None], ylim=[None,None],xlabels=None,ylabels=None):
        '''
        Plot the returns of multiple training runs as a line plot. 
        @param env_run_ids: list of run ids to plot
        @param labels: list of labels for the runs
        @param xsteps: if True, plot the returns at each step, otherwise plot the returns at each episode   
        @param smooth: if True, smooth the returns with a moving average
        @param save: if True, save the plot
        @param show: if True, show the plot
        @param plot_name: name of the plot
        @param sub_folder_name: name of the subfolder where to save the plot
        @param ext: extension of the plot
        @param xlim: x axis limits
        @param ylim: y axis limits
        @param xlabels: x axis labels
        @param ylabels: y axis labels
        '''
        if plot_name is None:
            plot_name = str(len(env_run_ids)) 

        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]  
        data = df_multiruns_episodes_returns(run_paths_list=run_paths_list, smooth=smooth, interpolate=xsteps, run_label_list=labels ) 
        save_path = os.path.join(self.save_multirun_training_plots_path, sub_folder_name, f"{plot_name}_multirun_returns_train.{ext}") 

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
 
    def multirun_returns_test(self, env_run_ids, labels=[], xlabels=None, ylabels=None,  save=True, show=True, plot_name=None, sub_folder_name="", plot_type="histplot", ext="pdf"): 
        '''
        Plot the returns of multiple testing runs as a line plot.
        @param env_run_ids: list of run ids to plot
        @param labels: list of labels for the runs
        @param xlabels: x axis labels
        @param ylabels: y axis labels
        @param save: if True, save the plot
        @param show: if True, show the plot
        @param plot_name: name of the plot
        @param sub_folder_name: name of the subfolder where to save the plot
        @param plot_type: type of plot to use
        @param ext: extension of the plot
        '''
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_test_folder, env_run) for env_run in env_run_ids]
        
        data = df_test_multirun_returns(run_paths_list=run_paths_list) 
        save_path = os.path.join(self.save_multirun_testing_plots_path, sub_folder_name, f"{plot_name}_{plot_type}_multirun_returns_test.{ext}")

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

 