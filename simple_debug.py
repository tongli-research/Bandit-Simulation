import numpy as np
import pandas as pd

import bayes_model as bm
import policy as pol
import sim_wrapper as sw


#################### test
#############################

hyperparams = {
    'n_rep':10, #number of replications
    'n_arm':3,
    'horizon': 180, # sample size
    'n_art_rep':2000, # no need for regular simulation
    'burn_in': 40,
    'batch_size': 1, #how often the bandit algorithm update itself based on new data
    'fast_batch_epsilon': 0, #a way to make simulation faster without hurting accuracy. if epsilon here >0, it means: at time T, it will update itself after T' = T+T*epsilon. Set epsilon =0 if wants to simulate and update at each single step
    'record_ap': False,  # True = forced estimation for all algortihm. False = depends on algorithm need
    'n_ap_rep': 200  # number of replications to approximate allocation probability
}
horizon = hyperparams['horizon']
n_arm = hyperparams['n_arm']
arr_dim = sw.ArrDim({'n_rep':0,'horizon':1,'n_arm':-1},hyperparams)

#mean = np.array([769, 1100, 620])/1500
#sd = np.array([15,15,15])/1500
# simple simulation
bandit = pol.StoBandit(reward_model=sw.RewardModel(model=np.random.binomial,
                                                   parameters={'n': [1,1,1], 'p': [0.5,0.5,0.5]})) #simulate the case where there's 3 arms, with Bernoulli reward mean = 0.5
res = sw.run_simulation(policy=bandit.ts_adapt_explor, #i.e. the PostDiff paper we are currently working on. We haven't exactly determine its for in multi-armed case. Here we use some naive version
                        algo_para = 0.3,
                        hyperparams=hyperparams)

#simulation result is saved in res
#check its method such as:
#res.reward_hist ; res.arm_counts ; etc