  
import numpy as np  
from mjrlenvs.scripts.tester import TestAgent

from mjrlenvs.run.pendulum_sac2 import Args

########################################################################

agent = "SAC_1_0"

tester = TestAgent(Args)
tester.loadmodel(agent)
# tester.registercallback()
mean, std = tester.evalpolicy(n_eval_episodes=5, render=False)
print(mean, std)
tester.plot(agent)
tester.infer()

########################################################################
########################################################################
########################################################################

########################################################################

# if not os.path.exists(PkgPath.testingdata()):
#     os.makedirs(PkgPath.testingdata())

# with open(PkgPath.testingdata("output.txt"), 'w') as f: 
#     line = "timestep,energy_tank,energy_exchanged" 
#     f.write(line)
#     f.close()

########################################################################

# def energycallback(locals_, _globals): 
#     energy_tank = locals_["info"]["energy_tank"]
#     print(energy_tank)
#     energy_exchanged = locals_["info"]["energy_exchanged"]
#     t = locals_["current_lengths"][0]
#     with open(PkgPath.testingdata("output.txt"), 'a') as f: 
#         line = f"\n{t},{energy_tank},{energy_exchanged}" 
#         f.write(line)
#         f.close()
  
