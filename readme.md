 
## Installation ##

Setup virtual environment

```
conda env create -f rl.yml
```


Add the following line to .bashrc

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/[...]/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```


```
pip install -e .
``` 


## Mujoco ##
## XML
quaternions are in the format [qw,qx,qy,qz]

### Contact Forces bug
Issue: https://github.com/openai/mujoco-py/pull/487
Need to modify 'mujoco_py/mjviewer.py' in the conda env as follow:
https://github.com/openai/mujoco-py/pull/487/commits/ab026c1ff8df54841a549cfd39374b312e8f00dd


## Usage ##

### Create an environment

- create a folder in 'data/envdesc' with the files 'arena.xml'(simulation model description) and a 'specs.json'(state and action specifics)
- create a class in 'mjrlenvs/envrl/' that extends 'EnvGymBase' or 'EnvGymGoalBase' ('from mjrlenvs.envrl.base import *' ) and implement here the observation and the reward computation 


### Run training

create a configuration file like 'some_run.py' in 'mjrlenvs/train/run' then 

'''
python mjrlenvs/run/some_run.py
'''

This can be either a single training or multiple training with same parametrs or a grid search. Import 'run()' function from 'trainer.py' if you want to use the same envisornment for exploration and evaluation, or import 'run()' from 'trainer2.py' if you want to use 2 different evnironments.

The best model, tensorboard output and a .txt file with parameters and mean reward can be found in 'mjrlenvs/data/testdata' 

## Usage ##

* fix trainer.py and runs with it
* fix panda