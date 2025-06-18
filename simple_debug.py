import numpy as np
import pandas as pd

import bayes_model as bm
import policy as pol
import sim_wrapper as sw


#################### test
#############################

hyperparams = sw.HyperParams(
    n_rep=5000,
    n_arm=3,
    horizon=250,
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
res = sw.run_simulation(policy=bandit.ts_postdiff_ur, #i.e. the PostDiff paper we are currently working on. We haven't exactly determine its for in multi-armed case. Here we use some naive version
                        algo_para = 0.3,
                        hyperparams=hyperparams)

#simulation result is saved in res
#check its method such as:
#res.reward_hist ; res.arm_counts ; etc