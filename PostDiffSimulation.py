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
import sys
from joblib import dump
#dump(variables, 'variables.joblib')
#loaded_variables = load('variables.joblib')


def critical_grid(h0_grid, h0_settings, h1_mean_reward):
    """
    :param h0_grid: 1 (n_rep) by horizon by len(h0_setting)
    :param h0_settings:
    :param h1_mean_reward: n_rep by horizon by 1
    :return:
    """
    mu_min = np.min(h0_settings)
    mu_max = np.max(h0_settings)
    grid_delta = (mu_max - mu_min) / (len(h0_settings) - 1)
    h1_mean_reward[h1_mean_reward<mu_min] = mu_min
    h1_mean_reward[h1_mean_reward>mu_max] = mu_max

    diff_arr = (h0_settings - h1_mean_reward)/grid_delta
    coef_arr = (1 - np.abs(diff_arr)) *(np.abs(diff_arr) < 1)

    return np.sum(coef_arr*h0_grid,axis=-1,keepdims=True)
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
    "eps_ts": [0, 1],
    "ts_postdiff_top": [0.125],
    #"ts_postdiff_ur": [0.125,0.15,0.2, 0.25],
    #"top2_ts": [0.6,0.7,0.85]
}

fpr_nulls = np.arange(0.15,0.86,0.1) #or 0.05, 0.95, 0.1

mu_list = [
    {'model':np.random.binomial, 'mu_center':[0.6, 0.5, 0.4], 'mu_scale': 0},#fixed scenario
    {'model':np.random.binomial, 'mu_center':[0.5, 0.5, 0.5], 'mu_scale': 0.1}, #cap at 0 and 1
    {'model':np.random.normal, 'mu_center':[0, 0, 0], 'mu_scale': 0.1}, #var fixed as 1
    #{'model':np.random.normal, 'mu_center':[0, 0, 0, 0, 0], 'mu_scale': 0.1}, #var fixed as 1
]

hyperparams = {
    'n_rep': 100,
    #'n_arm': 3, #can change to a list for loop later
    'horizon_per_arm': 800,  # should adapt based on number of arms
    'n_art_rep': 101,  # no need for regular simulation
    'burn_in': 6, #play each arm twice
    'batch_size': 1,
    "fast_batch_epsilon":0.1,
    'record_ap': False,  # True = forced estimation for all algorithm. False = depends on algorithm need
    'n_ap_rep': 100  # number of replications to approximate allocation probability
}

"""
##################################  Part 1.1: fixed parameters and duplicated variables  ########################################
"""
#now it needs to be changed for each sim
#horizon = hyperparams['horizon'] #for convenience
#n_arm = hyperparams['n_arm']
n_rep = hyperparams['n_rep']
arr_axis = {'n_rep': 0, 'horizon': 1, 'n_arm': -1} #for convience only. can move later

# action_history, the (i,j,k)-th element is 1 if in the i-th replication, at time step j, arm k is selected. 0 otherwise
# reward_history, the (i,j,k)-th element equal to reward if in the i-th replication, at time step j, arm k is selected.


"""
##################################  Part 2: simulation under H0  ########################################
output: critical values for all tests (ANOVA and Tukey) for all algorithm under h0 loop through 0.1 to 0.9
"""
algo_columns = [f"{algo} {param}" for algo, params in algorithms.items() for param in params]
columns = ['model', 'n_arm'] + algo_columns
h0_df = pd.DataFrame(columns=columns)
for mu in mu_list:
    model = mu['model']  # Get model name as string
    n_arm = len(mu['mu_center'])

    # Check if row already exists
    if not ((h0_df['model'] == model.__name__) & (h0_df['n_arm'] == n_arm)).any():
        # Create a new row with 'model', 'n_arm', and nulls for algo columns
        new_row = {col: None for col in h0_df.columns}
        new_row['model'] = model.__name__
        new_row['n_arm'] = n_arm
        for algo, params in algorithms.items():
            for para in params:
                tem_anova_res = []
                tem_tukey_res = []
                for mu_h0 in fpr_nulls:
                    if model.__name__ == 'binomial':
                        bandit = pol.StoBandit(reward_model=sw.RewardModel(model=model,
                                                                           parameters={'n': [1] * n_arm,
                                                                                       'p': [mu_h0] * n_arm}))

                        res = sw.run_simulation(policy=getattr(bandit, algo),
                                                algo_para=para,
                                                hyperparams=hyperparams)

                        tem_anova_res.append(np.quantile(res.anova(horizon=slice(None)),q=0.05, axis=arr_axis['n_rep']).flatten())
                        tem_tukey_res.append(np.quantile(np.max(res.tukey(horizon=slice(None)), axis=(-1, -2)),q=0.95, axis=arr_axis['n_rep']).flatten())

                    else:
                        bandit = pol.StoBandit(reward_model=sw.RewardModel(model=model,
                                                                           parameters={'loc': [mu_h0] * n_arm, #TODO: probably loc doesn't matter. remove it
                                                                                       'scale': 1})) #TODO: see if scale = other matters

                        res = sw.run_simulation(policy=getattr(bandit, algo),
                                                algo_para=para,
                                                hyperparams=hyperparams)

                        tem_anova_res.append(np.quantile(res.anova(horizon=slice(None)),q=0.05, axis=arr_axis['n_rep']).flatten())
                        tem_tukey_res.append(np.quantile(np.max(res.tukey(horizon=slice(None)), axis=(-1, -2)),q=0.95, axis=arr_axis['n_rep']).flatten())

                column_name = f"{algo} {para}"
                new_row[column_name] = {"anova":np.vstack(tem_anova_res).T[np.newaxis,:,:],
                                        "tukey":np.vstack(tem_tukey_res).T[np.newaxis,:,:],}
        h0_df = pd.concat([h0_df,  pd.DataFrame([new_row])], ignore_index=True)




#arms = [0, 1, 2]
#combinations = [(0, arm) for arm in arms if arm != 0]  # Get all order-specific combinations
# Lookup table for pairs
#len_combo = len(combinations)
# Example: Look up the result for a specific pair
#pair_to_lookup = (0, 1)
#index = pair_index.get(pair_to_lookup)

#work on H0 simulation separately later



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


power_constraint = 0.8


algo_columns = [f"{algo} {param}" for algo, params in algorithms.items() for param in params]
columns = ['model', 'n_arm'] + algo_columns
h1_df = pd.DataFrame(columns=columns)

ind = -1
for algo, params in algorithms.items():
    for para in params:
        ind += 1
        for i in range(len(mu_list)):
            np.random.seed(0)
            n_arm = len(mu_list[i]['mu_center'])
            horizon = hyperparams['horizon_per_arm']*n_arm
            true_reward_mean_dist = np.random.normal(scale=mu_list[i]['mu_scale'], loc=mu_list[i]['mu_center'], size=(n_rep, 1, n_arm))
            if mu_list[i]['model'].__name__ == 'binomial':
                true_reward_mean_dist[true_reward_mean_dist>0.99] = 0.99
                true_reward_mean_dist[true_reward_mean_dist<0.01] = 0.01
                reward_hist = np.random.binomial(n=1, p=true_reward_mean_dist, size=(n_rep, horizon, n_arm))
                bandit = pol.StoBandit(reward_model=sw.RewardModel(model=mu_list[i]['model'],
                                                                   parameters={'n': [1] * n_arm, 'p': [mu_h0] * n_arm}))
                res = sw.run_simulation(policy=getattr(bandit, algo),
                                        algo_para=para,
                                        hyperparams=hyperparams,
                                        reward_hist=reward_hist)

            else:
                reward_hist = np.random.normal(loc=true_reward_mean_dist, scale=1, size=(n_rep, horizon, n_arm))
                bandit = pol.StoBandit(reward_model=sw.RewardModel(model=np.random.normal,
                                                                   parameters={'loc': [mu_h0] * n_arm, 'scale': 1}))
                res = sw.run_simulation(policy=getattr(bandit, algo),
                                        algo_para=para,
                                        hyperparams=hyperparams,
                                        reward_hist=reward_hist)




            #power = np.mean(res.wald_test(horizon=slice(None)) > w_crit, axis=0)
            #step 0: find cirtical region in h0_df

            #Step 1: get the index for h0_df to match to the row with the same setting (number of arms & reward type)
            h0_index = ((h0_df['model'] == model.__name__) & (h0_df['n_arm'] == n_arm))

            #Step 2: get column index
            column_name = f"{algo} {para}"
            null_dist = h0_df[column_name][h0_index][1]

            anova_crit_values = critical_grid(null_dist['anova'], fpr_nulls, res.combined_means)
            tukey_crit_values = critical_grid(null_dist['tukey'], fpr_nulls, res.combined_means)


            #step 1: match anova power for power analysis
            anova_powers = np.mean(res.anova(slice(None))<anova_crit_values,axis = 0)


            tukey_result = res.tukey(slice(None))
            tukey_reshaped = tukey_result.reshape(*tukey_result.shape[:-2],-1)

            #the probability of testing positive between a random pair of arms
            tukey_power_pair_wise = np.sum(np.mean(tukey_reshaped > tukey_crit_values, axis=arr_axis['n_rep']), axis = -1)/ ( tukey_reshaped.shape[-1] -np.sqrt(tukey_reshaped.shape[-1]) )

            #The probability of having at least one pair being positive in a test
            tukey_power_testcase_wise = np.mean(np.max(tukey_reshaped > tukey_crit_values, axis=-1),axis=arr_axis['n_rep'])


            #calculate expected reward
            arm_means_n_exp = res_match.arm_means[:,n_exp,:]
            max_arm_ind = np.argmax(arm_means_n_exp, axis = 1)
            arm_selection = tukey_n_exp[np.arange(n_rep), max_arm_ind,:]<tukey_crit
            dpl_reward = np.mean(np.sum(power_analysis_dist * arm_selection, axis = 1)/np.sum(arm_selection,axis = 1))

            results_match.append({
                "algorithm": algo,
                "parameter": para,
                #"reward dist": p,
                "reward": np.mean(res_match.mean_reward,axis=0)[n_exp],
                "n_exp":n_exp,
                "anova": anova_powers[n_exp],
                "tukey": tukey_power,
                "dpl_reward": dpl_reward,
            })

            anova_powers = np.mean(res_dist.anova(slice(None)) < critical_grid(
                np.transpose(results_h0[ind]["anova_crit_arr"])[np.newaxis, :, ...], fpr_nulls, res_dist.mean_reward),
                                   axis=0)

            #n_exp = horizon - np.sum(anova_powers > power_constraint) - 1

            tukey_crit = critical_grid(np.transpose(results_h0[ind]["tukey_crit_arr"])[np.newaxis, :, ...], fpr_nulls,
                                       res_dist.mean_reward)[:, n_exp, :]

            tukey_n_exp = res_dist.tukey(n_exp)

            tukey_power = np.mean(np.max(tukey_n_exp > tukey_crit[..., np.newaxis], axis=(1, 2)))

            # calculate expected reward
            arm_means_n_exp = res_dist.arm_means[:, n_exp, :]
            max_arm_ind = np.argmax(arm_means_n_exp, axis=1)
            arm_selection = tukey_n_exp[np.arange(n_rep), max_arm_ind, :] < tukey_crit
            dpl_reward = np.mean(np.sum(true_reward_mean_dist[:,0,:] * arm_selection, axis=1) / np.sum(arm_selection, axis=1))

            results_dist.append({
                "algorithm": algo,
                "parameter": para,
                # "reward dist": p,
                "reward": np.mean(res_dist.mean_reward, axis=0)[n_exp],
                "n_exp": n_exp,
                "anova": anova_powers[n_exp],
                "tukey": tukey_power,
                "dpl_reward": dpl_reward,
            })


#pd.DataFrame(results_match).to_csv("resultsJan7th_match.csv", index=False)


dump(results_h0, 'results_h0_Dec26_v2.joblib')
dump({'results_dist':results_dist, 'results_match':results_match}, 'results_res_Dec26_v2.joblib')


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




#get H0_critical region
step_t = 799
stacked_array = np.stack([results_h0[i]['anova_crit_arr'][:, step_t] for i in range(len(results_h0))], axis=0)
#pd.DataFrame(stacked_array).to_csv("resultsJan9th_h0_t800.csv", index=False)

