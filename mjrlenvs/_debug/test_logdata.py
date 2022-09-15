from mjrlenvs.scripts.eval.logdata import LogData


logger = LogData()   
df = logger.df_runs_episodes_returns("/home/riccardo/projects/mjrlenvs/data/train/pendulum/prova3/logs", smooth=False)

print(df)