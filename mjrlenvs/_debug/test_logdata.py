from mjrlenvs.scripts.eval.logdata import *


df = df_run_episodes_returns("/home/riccardo/projects/mjrlenvs/data/train/pendulum/prova3/logs", smooth=False)

print(df)