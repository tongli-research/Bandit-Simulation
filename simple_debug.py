import numpy as np
import pandas as pd

import bayes_model as bm
import policy as pol
import sim_wrapper as sw


#################### test
#############################

hyperparams = {
    'n_rep':10,
    'n_arm':3,
    'horizon': 18, # sample size
    'n_art_rep':2000, # no need for regular simulation
    'burn_in': 4,
    'batch_size': 1,
    'record_ap': False,  # True = forced estimation for all algortihm. False = depends on algorithm need
    'n_ap_rep': 200  # number of replications to approximate allocation probability
}
horizon = hyperparams['horizon']
n_arm = hyperparams['n_arm']
arr_dim = sw.ArrDim({'n_rep':0,'horizon':1,'n_arm':-1},hyperparams)

mean = np.array([769, 1100, 620])/1500
sd = np.array([15,15,15])/1500
# simple simulation
bandit = pol.StoBandit(reward_model=sw.RewardModel(model=np.random.binomial,
                                                   parameters={'n': [1,1,1], 'p': [0.5,0.5,0.5]}),
                        arr_dim = arr_dim)
res0 = sw.run_simulation(policy=bandit.eps_top2_ts,
                        algo_para = 0.3,
                        hyperparams=hyperparams)