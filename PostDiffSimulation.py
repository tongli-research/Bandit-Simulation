import numpy as np
import pandas as pd

import bayes_model as bm
import policy as pol
import sim_wrapper as sw
from itertools import permutations
import copy
from scipy.special import logit
from scipy.special import expit
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
"""
##################################   Code documentation   ##################################

TODO Dec 24th
1. check if current TUkey match package result


Simulation for: 
1. test ART in UR
2. show that expected reward in deployment phase is similar for different algortihms
"""

"""
Note and todo:
1. see if 0.5 is a good guarantee (it seems not) and see if PostDiff is good (seems possible)
2. add normal simulation
3. maybe change here: policy=getattr(bandit, algo),
                                    algo_para=para, to align policy with algo_para
also: why we need 'getattr' again?

4. currently assume we only match power for ANOVA
5. what's the decision process??? assume there's a control (or not)?
if there's a control -> ANOVA fail means we choose the control?????????
or only have control in the 
control = 0.4. assume one is better than control -> control power of disconvering it

can we do that directly?

6. I think we don't use control:
    PostDiff doesn't seelct control seperately
    replace control sampling to mean sampling
X. for each algorithm:

draw setting 1:50
if seting = [0.6,0.4,0.3]? what's next? should we use 0.6+0.4+0.3 / 3 as FPR?
it's 
should we order setting p?

    conduct FPR estimation for pi==0.
    
    how to do it for Normal with unknown varaince???
#for binary and Normal
#how to match different algortihms? another level of search???
#for 'truth match reality', for different truth, ther's probably a differnt best parameter. How to account it?
#naive option: just choose one for all
# alternative 1: for epsilon-TS, do it for 0.25, 0.5, 0.75, and pick max for each, before taking average
# for PostDiff, design a meta-algortihm

############################   Part 0: High level plan   ##########################################
1.X set up inputs
2.X loop for each algortihm / potentially each n_arm to get Wald full distribution -> get step-wise critical region after
3.X code ANOVA / 
4. figure out the otehr test
5.0: create h1 simualtion environment
5. run loop in h1 case
6. calculate power and expected reward
7. 
Questions to think: 
1. how to tune parameters
2. how to choose 'n'
3. need define decesion process
"""

"""
##################################  Part 1: simulation settings  ########################################
"""

algorithms = {
    "eps_ts": [0, 1, 0.2, 0.4, 0.6],
    "ts_postdiff_top": [0.2, 0.25, 0.3],
    "ts_postdiff_ur": [0.2, 0.25, 0.3],
    "top2_ts": [0.7,0.85]
}

fpr_nulls = np.arange(0.25,0.75,0.1) #or 0.05, 0.95, 0.1

hyperparams = {
    'n_rep': 5000,
    'n_arm': 3, #can change to a list for loop later
    'horizon': 600,  # sample size
    'n_art_rep': 101,  # no need for regular simulation
    'burn_in': 6, #play each arm twice
    'batch_size': 1,
    'record_ap': False,  # True = forced estimation for all algorithm. False = depends on algorithm need
    'n_ap_rep': 100  # number of replications to approximate allocation probability
}

"""
##################################  Part 1.1: fixed parameters and duplicated variables  ########################################
"""
horizon = hyperparams['horizon'] #for convenience
n_arm = hyperparams['n_arm']
n_rep = hyperparams['n_rep']
arr_dim = sw.ArrDim({'n_rep': 0, 'horizon': 1, 'n_arm': -1}, hyperparams) #cannot change (make it fix later)


"""
##################################  Part 2: simulation under H0  ########################################
output: critical values for all tests (ANOVA and Tukey) for all algorithm under h0 loop through 0.1 to 0.9
"""
results_h0 = []
results = []

#arms = [0, 1, 2]
#combinations = [(0, arm) for arm in arms if arm != 0]  # Get all order-specific combinations
# Lookup table for pairs
#len_combo = len(combinations)
# Example: Look up the result for a specific pair
#pair_to_lookup = (0, 1)
#index = pair_index.get(pair_to_lookup)

#work on H0 simulation separately later

np.random.seed(1)

for algo, params in algorithms.items():
    """
    loop through null cases
    """

    for para in params:
        tem_res = []
        for mu_h0 in fpr_nulls:
            bandit = pol.StoBandit(reward_model=sw.RewardModel(model=np.random.binomial,
                                                               parameters={'n': [1] * n_arm, 'p': [mu_h0] * n_arm}),
                                   arr_dim=arr_dim)
            res = sw.run_simulation(policy=getattr(bandit, algo),
                                    algo_para=para,
                                    hyperparams=hyperparams)
            #tukey_res = res.tukey(horizon=slice(None))

            #save result for post hoc test
            tem_res.append({
                "anova_crit": np.quantile(res.anova(horizon = slice(None)),
                                          q=0.05,axis=arr_dim.arr_axis['n_rep']).flatten(),
                "tukey_crit": np.quantile(np.max(res.tukey(horizon=slice(None)),axis=(-1,-2)),
                                          q=0.95,axis=arr_dim.arr_axis['n_rep']).flatten(),
            })
        results_h0.append({
            "algorithm": algo,
            "parameter": para,
            "anova_crit_arr": np.vstack([d['anova_crit'] for d in tem_res]),
            "tukey_crit_arr": np.vstack([d['tukey_crit'] for d in tem_res]),
        })


            # save it in a dict
            # maybe consider also control FPR for pairwise post hoc test???

"""
##################################  Part 3: simulation under H1  ########################################
"""

"""
there will be 2 settings.
q: do we want the control?

setting 1:



setting 2:
assume power analysis in: [0.6, 0.4 ... 0.4]
actual mus follow N(mu_ori , scale = 0.5?)

scale = 1 is a little too big..

maybe try [0.6,0.5,0.4,0.3,0.2] and a normal one

mu_control = 0.5? maybe try 3 values (with sigma fixed to be 0.3?)

logit score -> 0 

for all rest mu, sample from N(0, sigma)  ,  try maybe 2 sigma values ( 0.3 and  1?)

for sigma = 1 -> can easily get 0.85 ~ 0.9 
for scale = 0.3 -> result around 0.4 - 0.6 (make sense for sample size 200)


plan:
case 1: mu (loc) = 0.5, scale = 1
case 2: mu (loc) = 0.8, scale = 0.3
case 3: mu (loc) = 0.5, scale = 0.3
case 4: mu (loc) = 0.2, scale = 0.3
"""

"""
##################################  Part 3.1: generate problem instances  ########################################
"""

#just try one setting for now
#set seed
reward_dist = [
    [0.6]+[0.5]*(n_arm-2)+[0.4], #what's the relation to control ?
]
mu_control = [0.5, 0.8, 0.5, 0.2]
mu_scale = [1, 0.3, 0.3, 0.3]

np.random.seed(0)
true_reward_mean_dist = np.random.normal(scale=0.05,loc=0.5,size=(n_rep,1,n_arm))
reward_hist = np.random.binomial(n=1,p=true_reward_mean_dist,size=(n_rep,horizon,n_arm))
for algo, params in algorithms.items():
    for para in params:
        for i in range(len(mu_control)):
            np.random.seed(0)

            bandit = pol.StoBandit(
                reward_model=sw.RewardModel(model=np.random.binomial, parameters={'n': [1]*hyperparams['n_arm'], 'p': p}),
                arr_dim=arr_dim)
            res_match = sw.run_simulation(policy=getattr(bandit, algo),
                                          algo_para=para,
                                        hyperparams=hyperparams)
            res_dist = sw.run_simulation(policy=getattr(bandit, algo),
                                    algo_para=para,
                                    hyperparams=hyperparams,
                                    reward_hist=reward_hist)
            #power = np.mean(res.wald_test(horizon=slice(None)) > w_crit, axis=0)

            # Compute the mean for each combination and store in an array
            power_array = np.vstack([
                np.mean(res.wald_test(arm1_index=arm1, arm2_index=arm2, horizon=slice(None)) > w_crit, axis=0)
                for arm1, arm2 in combinations
            ])

            results.append({
                "algorithm": algo,
                "parameter": para,
                "reward dist": p,
                "reward": res.mean_reward,
                "power": power_array,
            })


"""
Part 3: Save results
"""
df_results = pd.DataFrame(results)
df_results[['p1', 'p2', 'p3']] = pd.DataFrame(df_results['reward dist'].tolist(), index=df_results.index)


df_results.to_csv("resultsDec4_pd1.csv", index=False)

"""
Process results
"""

#for each algorithm, determine their n_exp
pull_power_list = df_results.loc[df_results['reward dist'].apply(lambda x: x == [0.6, 0.4, 0.4]), 'power'].to_numpy()
power_array_h1 = np.vstack([arr[0] for arr in pull_power_list])
n_exp = np.sum(power_array_h1 < 0.8,axis=1)
algo_length = sum(len(v) for v in algorithms.values())
#calculate their power in other settings
processed_res_list = []
for p in reward_dists:
    pull_power_list = df_results.loc[df_results['reward dist'].apply(lambda x: x == p), 'power'].to_numpy()
    tem_power_array = np.vstack(pull_power_list)

    pull_reward_list = df_results.loc[df_results['reward dist'].apply(lambda x: x == p), 'reward'].to_numpy()
    tem_reward_array = np.vstack(pull_reward_list)

    processed_res_list.append({
        "setting": p,
        'power_l': tem_power_array[np.arange(algo_length* (n_arm-1) ), np.repeat(n_exp-1, (n_arm-1))],
        'reward_l': tem_reward_array[np.arange(algo_length), n_exp-1]
    })
processed_res_df = pd.DataFrame(processed_res_list)
power_array = np.transpose(np.vstack(processed_res_df['power_l']))
reward_array = np.transpose(np.vstack(processed_res_df['reward_l']))

columns = [f"p1={p1}, p2={p2}, p3={p3}" for p1, p2, p3 in reward_dists]
rows = [f"{algo}, {param}" for algo, params in algorithms.items() for param in params]


#calcualte expected reward
reward_dist_array = pd.DataFrame(reward_dists).to_numpy()
p_arm1 = power_array[np.arange(algo_length)*(n_arm-1),:]
p_arm12 = power_array[(np.arange(algo_length)*(n_arm-1)+1),:]
reward_dpl = p_arm1* reward_dist_array[:,0] + (p_arm12-p_arm1) * (reward_dist_array[:,0]+reward_dist_array[:,1])/2 + (1- p_arm12) * np.mean(reward_dist_array,axis=1)

# Creating a DataFrame with NaN values
df_reward_dpl = pd.DataFrame(np.stack(reward_dpl,np.transpose(n_exp)),index=rows, columns=columns)
df_reward_dpl.to_csv("dec4_match_exp_reward_v3.csv")

df_power = pd.DataFrame(power_array)
df_power.to_csv("dec4_power3.csv")

pd.DataFrame(reward_array).to_csv("dec4_reward3.csv")
