"""
testing the correctness of implementation of TS-PostDiff and epsilon-TS, based on previous results
Ground truth:
horizon/sample size = 197, p1=0.6, p2=0.4
1. for 'c' = 0.175:               Power = 0.77(0), Reward = 0.550
2. for 'eps' = 0.3 / beta = 0.85: Power = 0.70(2), Reward = 0.552
"""

import numpy as np
import pandas as pd

import bayes_vector_ops as bm
import policy as pol
import sim_wrapper as sw

"""
Simulation for: 
1. test ART in UR
2. show that expected reward in deployment phase is similar for different algortihms
"""


hyperparams = sw.SimulationConfig(
    n_rep=10000,
    n_arm=2,
    horizon=197,
    burn_in=2,
    base_batch_size= 1,
    batch_scaling_rate=None,
)

horizon = hyperparams.horizon
n_arm = hyperparams.n_arm

# simple simulation
bandit = pol.StoBandit(reward_model=sw.RewardModel(model=np.random.binomial, parameters={'n': [1, 1], 'p': [0.6, 0.4]}))


res = sw.run_simulation(policy=getattr(bandit, 'eps_ts'),
                        algo_para=0.3,
                        sim_config=hyperparams)
res_power = np.mean(np.abs(res.wald_test())>1.96)



res1 = sw.run_simulation(policy=getattr(bandit, 'ts_postdiff_ur'),
                         algo_para=0.175,
                         sim_config=hyperparams)
res1_power = np.mean(np.abs(res1.wald_test())>1.96)

print(f"Reward for epsilon-TS (0.3) is: {res.mean_reward[196]:.4f}, (should be 0.552)")
print(f"Power for epsilon-TS (0.3) is: {res_power:.4f}, (should be 0.702)")

print(f"Reward for TS-PostDiff (0.175) is: {res1.mean_reward[196]:.4f}, (should be 0.550)")
print(f"Power for TS-PostDiff (0.175) is: {res1_power:.4f}, (should be 0.766)")


def test_reward_matches_baseline():
    assert abs(res.mean_reward[196] - 0.552) < 0.02