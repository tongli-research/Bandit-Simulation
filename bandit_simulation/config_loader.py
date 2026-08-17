"""
YAML-based configuration loader for bandit simulation experiments.

Loads a YAML config file and produces:
  - A base SimulationConfig
  - A list of sweep_specs compatible with sweep_and_run()

Usage:
    from bandit_simulation.config_loader import load_config
    base_config, sweep_specs = load_config("configs/table4.yaml")
"""

import numpy as np
import yaml

from . import bandit_algorithm as algorithm
from .simulation_configurator import SimulationConfig
from .test_procedure_configurator import ANOVA, TConstant, TControl, Tukey

# ── Registries ──────────────────────────────────────────────────────────────

TEST_PROCEDURE_REGISTRY = {
    "ANOVA": ANOVA,
    "TConstant": TConstant,
    "TControl": TControl,
    "Tukey": Tukey,
}

ALGORITHM_REGISTRY = {
    "EpsTS": algorithm.EpsTS,
    "TSProbClip": algorithm.TSProbClip,
    "TSPostDiff": algorithm.TSPostDiff,
    "TSPostDiffLinear": algorithm.TSPostDiffLinear,
    "TSLinearWC": algorithm.TSPostDiffLinear,  # backwards compat alias, renamed class
    "RoundRobin": algorithm.RoundRobin,
    "EpsGreedy": algorithm.EpsGreedy,
    "LinearUCB": algorithm.LinearUCB,
    "UCB": algorithm.UCB,
}

REWARD_MODEL_MAP = {
    "binomial": np.random.binomial,
    "bernoulli": np.random.binomial,
    "normal": np.random.normal,
    "gaussian": np.random.normal,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_param_list(spec):
    """Convert shorthand param list specs to actual Python lists.

    Supports:
      - Plain list: [0.0, 0.5, 1.0]
      - linspace: {linspace: [start, stop, num]}
      - arange:   {arange: [start, stop, step]}
    """
    if isinstance(spec, list):
        return [float(v) for v in spec]
    if isinstance(spec, dict):
        if "linspace" in spec:
            args = spec["linspace"]
            return list(map(float, np.linspace(*args)))
        if "arange" in spec:
            args = spec["arange"]
            return list(map(float, np.arange(*args)))
    raise ValueError(f"Cannot parse param list spec: {spec}")


def _build_test_procedure(spec):
    """Build a TestProcedure object from a YAML dict.

    Example spec:
        {type: ANOVA}
        {type: TConstant, params: {test_type: one-sided, min_effect: 0.1}}
    """
    tp_name = spec["type"]
    if tp_name not in TEST_PROCEDURE_REGISTRY:
        raise ValueError(
            f"Unknown test procedure '{tp_name}'. "
            f"Available: {list(TEST_PROCEDURE_REGISTRY.keys())}"
        )
    cls = TEST_PROCEDURE_REGISTRY[tp_name]
    params = spec.get("params", {})
    return cls(**params)


def _resolve_algorithms(spec):
    """Resolve algorithm name strings to classes."""
    if isinstance(spec, str):
        spec = [spec]
    result = []
    for name in spec:
        if name not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{name}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
        result.append(ALGORITHM_REGISTRY[name])
    return result


# ── Main loader ─────────────────────────────────────────────────────────────

def load_config(yaml_path):
    """Load a YAML config file and return (base_config, sweep_specs).

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.

    Returns
    -------
    base_config : SimulationConfig
        The base simulation configuration.
    sweep_specs : list[dict]
        Sweep specifications for sweep_and_run().
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    sim_sec = cfg.get("simulation", {})
    test_sec = cfg.get("test_procedure", {})
    sweep_sec = cfg.get("sweep", {})

    # ── Build base test procedure ───────────────────────────────────
    base_test_proc = _build_test_procedure(test_sec) if test_sec else ANOVA()

    # ── Build reward model ──────────────────────────────────────────
    reward_model_name = sim_sec.get("reward_model", "binomial")
    reward_model = REWARD_MODEL_MAP.get(reward_model_name)
    if reward_model is None:
        raise ValueError(
            f"Unknown reward model '{reward_model_name}'. "
            f"Available: {list(REWARD_MODEL_MAP.keys())}"
        )

    # ── Build base SimulationConfig ─────────────────────────────────
    config_kwargs = dict(
        n_rep=sim_sec.get("n_rep", 10000),
        n_arm=sim_sec.get("n_arm", 2),
        horizon=sim_sec.get("horizon", 1000),
        burn_in_per_arm=sim_sec.get("burn_in_per_arm", 1),
        reward_model=reward_model,
        reward_evaluation_method=sim_sec.get("reward_evaluation_method", "reward"),
        test_procedure=base_test_proc,
    )

    # arm_mean_reward_dist_spec
    arm_dist = sim_sec.get("arm_mean_reward_dist")
    if arm_dist:
        config_kwargs["arm_mean_reward_dist_spec"] = {
            "dist": arm_dist["dist"],
            "params": arm_dist.get("params", {}),
        }

    # Optional fields
    if "reward_std" in sim_sec:
        config_kwargs["reward_std"] = sim_sec["reward_std"]
    if "base_batch_size" in sim_sec:
        config_kwargs["base_batch_size"] = sim_sec["base_batch_size"]
    if "batch_scaling_rate" in sim_sec:
        config_kwargs["batch_scaling_rate"] = sim_sec["batch_scaling_rate"]

    base_config = SimulationConfig(**config_kwargs)

    # ── Build sweep specs ───────────────────────────────────────────
    sweep_specs = []

    # Algorithms
    algo_spec = sweep_sec.get("algo")
    if algo_spec:
        sweep_specs.append({"algo": _resolve_algorithms(algo_spec)})

    # Algorithm parameters
    param_spec = sweep_sec.get("algo_param_list")
    if param_spec:
        sweep_specs.append({"algo_param_list": _resolve_param_list(param_spec)})

    # Test procedure sweep (optional)
    tp_list = sweep_sec.get("test_procedure")
    if tp_list:
        sweep_specs.append({
            "test_proc": [_build_test_procedure(tp) for tp in tp_list]
        })

    # Arm mean reward distribution sweep (optional, e.g. Table 5 mis-specification)
    arm_dist_list = sweep_sec.get("arm_mean_reward_dist")
    if arm_dist_list:
        sweep_specs.append({
            "arm_mean_reward_dist_spec": [
                {"dist": d["dist"], "params": d.get("params", {})}
                for d in arm_dist_list
            ]
        })

    # Horizon sweep (optional)
    horizon_spec = sweep_sec.get("horizon")
    if horizon_spec:
        if isinstance(horizon_spec, list):
            sweep_specs.append({"horizon": horizon_spec})

    return base_config, sweep_specs
