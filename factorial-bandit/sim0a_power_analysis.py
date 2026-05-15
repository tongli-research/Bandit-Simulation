"""
Sim 0a: Power analysis for 2x3 factorial design under uniform allocation.
Analytical computation + Monte Carlo verification.

No simulation framework dependency -- pure numpy/scipy.

Output: prints MDE table and verification (no file output; results in simulation_plan.md)
"""
import numpy as np
from scipy import stats

F = np.array([
    [1, 0, 0], [1, 1, 0], [1, 0, 1],
    [1, 1, 1], [1, 0, 2], [1, 1, 2],
])
K = 6
SIGMA = 0.5
ALPHA = 0.05
POWER_TARGET = 0.80
TARGET_MDE = 0.10
N_MC_REP = 50_000

FtF = F.T @ F
FtF_inv = np.linalg.inv(FtF)

z_alpha2 = stats.norm.ppf(1 - ALPHA / 2)
z_power = stats.norm.ppf(POWER_TARGET)
z_sum = z_alpha2 + z_power


def mde_at_T(j, T):
    return z_sum * SIGMA * np.sqrt(K * FtF_inv[j, j] / T)


def power_at_T(j, T, beta_j):
    se = SIGMA * np.sqrt(K * FtF_inv[j, j] / T)
    return (1 - stats.norm.cdf(z_alpha2 - beta_j / se)
            + stats.norm.cdf(-z_alpha2 - beta_j / se))


def required_T(j, mde):
    return int(np.ceil(z_sum ** 2 * SIGMA ** 2 * K * FtF_inv[j, j] / mde ** 2))


def main():
    print("Analytical Power Analysis: 2x3 Factorial Under UR")
    print(f"sigma={SIGMA}, alpha={ALPHA}, power={POWER_TARGET}")
    print(f"\n(F'F)^-1 diagonal: beta_1={FtF_inv[1,1]:.4f}, beta_2={FtF_inv[2,2]:.4f}")
    print(f"Variance ratio: beta_1 is {FtF_inv[1,1]/FtF_inv[2,2]:.2f}x beta_2")

    T1 = required_T(1, TARGET_MDE)
    T2 = required_T(2, TARGET_MDE)
    print(f"\nRequired T for MDE={TARGET_MDE}: beta_1 -> {T1}, beta_2 -> {T2}")
    print(f"Both factors: T = {max(T1, T2)}")

    print(f"\n{'T':>6}  {'MDE_1':>7}  {'MDE_2':>7}  {'Pow_1@0.1':>10}  {'Pow_2@0.1':>10}")
    for T in [200, 400, 600, T2, T1, 1000, 1500, 2000]:
        print(f"{T:>6}  {mde_at_T(1,T):>7.3f}  {mde_at_T(2,T):>7.3f}  "
              f"{power_at_T(1,T,0.1):>9.1%}  {power_at_T(2,T,0.1):>9.1%}")

    print(f"\nMonte Carlo verification ({N_MC_REP} reps)...")
    rng = np.random.default_rng(42)
    beta_true = np.array([0.30, 0.10, 0.10])
    arm_means = F @ beta_true

    for T in [T2, T1]:
        n_per_arm = T // K
        T_actual = n_per_arm * K
        X = np.tile(F, (n_per_arm, 1))
        XtX_inv = np.linalg.inv(X.T @ X)
        XtX_inv_Xt = XtX_inv @ X.T

        y_mean = np.tile(arm_means, n_per_arm)
        Y = y_mean[np.newaxis, :] + rng.normal(0, SIGMA, size=(N_MC_REP, T_actual))
        beta_hat = (XtX_inv_Xt @ Y.T).T

        Y_hat = (X @ beta_hat.T).T
        sigma_hat_sq = np.sum((Y - Y_hat) ** 2, axis=1) / (T_actual - 3)
        se = np.sqrt(sigma_hat_sq[:, np.newaxis] * np.diag(XtX_inv)[np.newaxis, :])
        t_stats = beta_hat / se
        crit = stats.t.ppf(1 - ALPHA / 2, df=T_actual - 3)

        mc_pow1 = np.mean(np.abs(t_stats[:, 1]) > crit)
        mc_pow2 = np.mean(np.abs(t_stats[:, 2]) > crit)
        an_pow1 = power_at_T(1, T_actual, 0.1)
        an_pow2 = power_at_T(2, T_actual, 0.1)

        print(f"  T={T_actual}: analytical ({an_pow1:.1%}, {an_pow2:.1%}) "
              f"vs MC ({mc_pow1:.1%}, {mc_pow2:.1%})")


if __name__ == "__main__":
    main()
