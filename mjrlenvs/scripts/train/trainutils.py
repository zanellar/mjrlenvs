
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
 
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        # print(progress_remaining * initial_value)
        return progress_remaining * initial_value

    return func
  
class SaveTrainingConfigurations():

    def __init__(self, run_results_file_path, args): 
        self.run_results_file_path = run_results_file_path

        with open(self.run_results_file_path, 'w') as file:
            self.args = args
            line = "agent,"
            for k in self.args.AGENT_PARAMS.keys():
                line += k + ","
            line += "mean,std"  
            file.write(line) 
    
    def add(self, name, config, res):   
        with open(self.run_results_file_path, 'a') as file:
            line = f"\n{name},"
            for v in config.values():
                line += str(v) + "," 
            line += f"{res[0]},{res[1]}"   
            file.write(line) 
  