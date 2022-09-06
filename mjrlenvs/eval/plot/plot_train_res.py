from numpy import load
from mjrlenvs.scripts.pkgpaths import PkgPath

path = PkgPath.trainingdata('pendulum/baseline_r36_cp/checkpoints/SAC_1_0/eval_results/evaluations.npz')
data = load(path) 
lst = data.files
for item in lst:
    print(item)
    print(data[item])
