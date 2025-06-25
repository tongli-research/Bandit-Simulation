import numpy as np
import pandas as pd

import bayes_model as bm
import policy as pol
import sim_wrapper as sw


#################### test
#############################
#for vm?
import sys
import os

# Add project root (where bayes_model.py is) to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

hyperparams = sw.HyperParams(
    n_rep=10000,
    n_arm=3,
    horizon=1000,
    burn_in=5,
    batch_size = None,
    fast_batch_epsilon = 0.1,
)

horizon = hyperparams.horizon
n_arm = hyperparams.n_arm
arr_dim = sw.ArrDim({'n_rep':0,'horizon':1,'n_arm':-1},hyperparams)

#mean = np.array([769, 1100, 620])/1500
#sd = np.array([15,15,15])/1500
# simple simulation
bandit = pol.StoBandit(reward_model=sw.RewardModel(model=np.random.binomial,
                                                   parameters={'n': [1,1,1], 'p': [0.6,0.5,0.5]})) #simulate the case where there's 3 arms, with Bernoulli reward mean = 0.5
import time

start_time = time.time()

for i in range(5):
    res1 = sw.run_simulation(
        policy=bandit.ts_postdiff_ur,
        algo_para=0.3,
        hyperparams=hyperparams
    )

end_time = time.time()

print(f"Total time for 10 runs: {end_time - start_time:.2f} seconds")


#simulation result is saved in res
#check its method such as:
#res.reward_hist ; res.arm_counts ; etc