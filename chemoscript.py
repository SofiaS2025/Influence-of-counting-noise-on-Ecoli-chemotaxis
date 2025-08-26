import sys
import numpy as np
import subprocess

Ntraj=120 # 120 bacteria / trajectories

for ii in range(Ntraj):
    subprocess.call(['python3','main.py',str(ii)]) 
    sys.stdout.flush() 
