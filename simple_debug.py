import numpy as np
import pandas as pd
import time
import bayes_model as bm
import policy as pol
import sim_wrapper as sw
from joblib import Parallel, delayed

#################### test
#############################
#for vm?
import sys
import os

# Add project root (where bayes_model.py is) to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

hyperparams = sw.HyperParams(
    n_rep=400,
    n_arm=3,
    horizon=10000,
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
params_dict = {'policy':bandit.ts_postdiff_ur,'algo_para':0.3, 'hyperparams':hyperparams}

n_loop = 10

start_time = time.time()

for i in range(n_loop):
    res1 = sw.run_simulation(
        policy=bandit.ts_postdiff_ur,
        algo_para=0.3,
        hyperparams=hyperparams
    )

end_time = time.time()

print(f"Total time for regular: {end_time - start_time:.2f} seconds")

start_time = time.time()
Parallel(n_jobs=-1)(delayed(sw.run_simulation)(**params_dict) for _ in range(n_loop))

end_time = time.time()

print(f"Total time for paraller: {end_time - start_time:.2f} seconds")
#simulation result is saved in res
#check its method such as:
#res.reward_hist ; res.arm_counts ; etc