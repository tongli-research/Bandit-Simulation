import json
import os
import time
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, redirect, jsonify
from waitress import serve

from bandit_simulation import ANOVA, EpsTS, TConstant, TControl, TSProbClip, TSTopUR, Tukey
from recommendation import (
    get_recommendation, get_recommendation_adaptive,
    create_simulation_config, run_single_check,
)

app = Flask(__name__)

SCENARIOS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_scenarios.json')
MAX_SCENARIOS = 30

# Global list of completed results during simulation (for live results table)
completed_results = []

# Cancel flag (checked between configs)
cancel_requested = False

# Global simulation progress (single-user app)
sim_progress = {
    'running': False,
    'completed': 0,
    'total': 0,
    'algo_name': '',
    'algo_param': 0.0,
    'start_time': 0,
    'completed_elapsed': 0,
    'current_horizon': 0,
    'max_horizon': 0,
    'current_power': None,
    'current_phase': '',
    'batch_progress': 0,
}


def load_scenarios():
    if os.path.exists(SCENARIOS_FILE):
        with open(SCENARIOS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_scenarios_file(scenarios):
    with open(SCENARIOS_FILE, 'w') as f:
        json.dump(scenarios, f, indent=2)


REWARD_MODEL_MAP = {
    'bernoulli': np.random.binomial,
    'gaussian': np.random.normal,
}

ALGORITHM_MAP = {
    "TSProbClip": TSProbClip,
    "EpsTS": EpsTS,
    "TSTopUR": TSTopUR,
}

AVAILABLE_ALGOS = list(ALGORITHM_MAP.keys())


def build_test_procedure(sp):
    """Build a test procedure object from a sim_params dict."""
    test_name = sp['test_name']
    kwargs = dict(
        type1_error_constraint=sp['type1_error'],
        power_constraint=sp['test_const'],
        min_effect=sp['min_effect'],
        family_wise_error_control=sp['family_wise_error_control'],
    )
    if test_name == 'anova':
        return ANOVA(**kwargs)
    elif test_name == 't_control':
        test_type = 'one-sided' if sp.get('is_one_tail_control') == 'one-sided' else 'two-sided'
        return TControl(**kwargs, test_type=test_type,
                        control_group_index=sp.get('t_control_param', 0))
    elif test_name == 't_constant':
        test_type = 'one-sided' if sp.get('is_one_tail_const') == 'one-sided' else 'two-sided'
        return TConstant(**kwargs, test_type=test_type,
                         constant_threshold=sp.get('t_constant_param', 0.5))
    else:
        tukey_type = sp.get('tukey_test_type', 'distinct-best-arm')
        if tukey_type not in ('all-pair-wise', 'distinct-best-arm'):
            tukey_type = 'distinct-best-arm'
        return Tukey(**kwargs, test_type=tukey_type)


@app.route('/')
@app.route('/index')
def index():
    scenarios = load_scenarios()
    scenario_list = [
        {
            'name': name,
            'created_at': data.get('created_at', ''),
            'is_default': data.get('is_default', False),
        }
        for name, data in scenarios.items()
    ]
    return render_template('index.html', saved_scenarios=scenario_list)


@app.route('/scenario/<name>')
def load_scenario(name):
    scenarios = load_scenarios()
    if name not in scenarios:
        return redirect('/')

    scenario = scenarios[name]
    chart_data_json = json.dumps(scenario['chart_data'])

    results_summary = {
        'test_procedure': scenario.get('test_procedure', ''),
        'n_algorithms_tested': scenario.get('n_algorithms_tested', 0),
        'n_parameter_values': scenario.get('n_parameter_values', 0),
        'chart_data_json': chart_data_json,
    }

    sp = scenario.get('sim_params') or {}
    # Backward compatibility: generate algo_param_config from algo_names if missing
    if 'algo_param_config' not in sp and 'algo_names' in sp:
        sp['algo_param_config'] = {
            n: {"min": 0.0, "max": 1.0, "granularity": 21}
            for n in sp['algo_names']
        }
    power_constraint = sp.get('test_const', 0.80) if sp else 0.80

    return render_template(
        'recommend.html',
        user_inputs=scenario.get('inputs', {}),
        recommendations=[],
        plot_path=None,
        results_summary=results_summary,
        chart_data_json=chart_data_json,
        sim_params=sp,
        max_reward=None,
        power_constraint=power_constraint,
        n_h0_cores=sp.get('n_h0_cores'),
        n_h0_reps_per_core=sp.get('n_h0_reps'),
        available_algos=AVAILABLE_ALGOS,
    )


@app.route('/scenario/<name>/params')
def scenario_params(name):
    """Return sim_params and inputs for a saved scenario (for form pre-fill)."""
    scenarios = load_scenarios()
    if name not in scenarios:
        return jsonify({'error': 'Scenario not found'}), 404
    scenario = scenarios[name]
    sp = scenario.get('sim_params') or {}
    # Backward compatibility: generate algo_param_config from algo_names if missing
    if 'algo_param_config' not in sp and 'algo_names' in sp:
        sp['algo_param_config'] = {
            name: {"min": 0.0, "max": 1.0, "granularity": 21}
            for name in sp['algo_names']
        }
    return jsonify({
        'sim_params': sp,
        'inputs': scenario.get('inputs', {}),
    })


@app.route('/save_scenario', methods=['POST'])
def save_scenario():
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        name = datetime.now().strftime('%Y-%m-%d_%H:%M')

    scenarios = load_scenarios()

    if len(scenarios) >= MAX_SCENARIOS and name not in scenarios:
        return jsonify({'error': f'Maximum {MAX_SCENARIOS} scenarios reached. Delete some first.'}), 400

    scenarios[name] = {
        'inputs': data.get('inputs', {}),
        'chart_data': data.get('chart_data', []),
        'test_procedure': data.get('test_procedure', ''),
        'n_algorithms_tested': data.get('n_algorithms_tested', 0),
        'n_parameter_values': data.get('n_parameter_values', 0),
        'sim_params': data.get('sim_params'),
        'created_at': datetime.now().isoformat(),
        'is_default': False,
    }

    save_scenarios_file(scenarios)
    return jsonify({'success': True, 'name': name})


@app.route('/delete_scenario/<name>', methods=['POST'])
def delete_scenario(name):
    scenarios = load_scenarios()
    if name in scenarios:
        if scenarios[name].get('is_default', False):
            return jsonify({'error': 'Cannot delete default scenarios.'}), 400
        del scenarios[name]
        save_scenarios_file(scenarios)
    return jsonify({'success': True})


@app.route('/progress')
def progress():
    """Return current simulation progress for the loading overlay."""
    p = sim_progress.copy()
    if p['running'] and p['start_time'] > 0:
        p['elapsed'] = round(time.time() - p['start_time'], 1)
        if p['completed'] > 0 and p['total'] > 0:
            # avg uses time at last completion (not current elapsed)
            avg_per_config = p['completed_elapsed'] / p['completed']
            est_total = avg_per_config * p['total']
            p['est_remaining'] = round(max(est_total - p['elapsed'], 0), 1)
        else:
            p['est_remaining'] = None
    else:
        p['elapsed'] = 0
        p['est_remaining'] = None
    return jsonify(p)


@app.route('/progress_results')
def progress_results():
    """Return completed simulation results for the live results table."""
    return jsonify(completed_results)


@app.route('/cancel', methods=['POST'])
def cancel():
    """Signal the running simulation to stop after the current config."""
    global cancel_requested
    cancel_requested = True
    return jsonify({'success': True})


@app.route('/recommend', methods=["post"])
def recommend():
    print("Form submitted!")

    # --- Parse core simulation parameters ---
    n_arm = int(request.form['n_arm'])
    horizon = int(request.form['horizon'])
    n_rep = int(request.form['n_rep'])

    reward_distribution = request.form['reward_distribution']
    reward_model = REWARD_MODEL_MAP[reward_distribution]
    reward_std = float(request.form['reward_std']) if reward_distribution == 'gaussian' else None

    h1_loc = float(request.form['h1_loc'])
    h1_scale = float(request.form['h1_scale'])
    max_reward_str = request.form.get('max_reward', '').strip()
    max_reward = float(max_reward_str) if max_reward_str else None

    # --- Parse statistical test parameters ---
    test_name = request.form['test_name']
    type1_error = float(request.form['type1_error_constraint'])
    test_const = float(request.form['test_const'])
    min_effect = float(request.form['min_effect'])
    family_wise_error_control = request.form.get('family_wise_error_control') == 'on'

    t_control_param = int(request.form.get('t_control_param', 0))
    t_constant_param = float(request.form.get('t_constant_param', 0.0))
    is_one_tail_control = request.form.get('is_one_tail_control', None)
    is_one_tail_const = request.form.get('is_one_tail_const', None)
    tukey_test_type: str | None = request.form.get('tukey_test_type', None)

    # --- Build sim_params (raw values for re-running simulations) ---
    sim_params = {
        'n_arm': n_arm, 'horizon': horizon, 'n_rep': n_rep,
        'reward_distribution': reward_distribution,
        'h1_loc': h1_loc, 'h1_scale': h1_scale, 'reward_std': reward_std,
        'test_name': test_name, 'type1_error': type1_error,
        'test_const': test_const, 'min_effect': min_effect,
        'family_wise_error_control': family_wise_error_control,
        't_control_param': t_control_param,
        't_constant_param': t_constant_param,
        'is_one_tail_control': is_one_tail_control,
        'is_one_tail_const': is_one_tail_const,
        'tukey_test_type': tukey_test_type,
    }

    # --- Parse optional H0 config ---
    n_h0_cores_str = request.form.get('n_h0_cores', '').strip()
    n_h0_reps_str = request.form.get('n_h0_reps', '').strip()
    sim_params['n_h0_cores'] = int(n_h0_cores_str) if n_h0_cores_str else None
    sim_params['n_h0_reps'] = int(n_h0_reps_str) if n_h0_reps_str else None

    # --- Create test procedure ---
    test_procedure = build_test_procedure(sim_params)

    # Apply user-specified H0 config overrides
    if sim_params['n_h0_cores']:
        # In linear mode, n_crit_sim_groups + 1 = n_cores, so subtract 1
        if test_procedure.n_crit_approx_method == 'linear':
            test_procedure.n_crit_sim_groups = max(sim_params['n_h0_cores'] - 1, 1)
        else:
            test_procedure.n_crit_sim_groups = sim_params['n_h0_cores']
    if sim_params['n_h0_reps']:
        test_procedure.n_crit_sim_rep = sim_params['n_h0_reps']

    # --- Build user inputs summary for display ---
    user_inputs = {
        'Number of Arms': n_arm,
        'Horizon': horizon,
        'Number of Repetitions': n_rep,
        'Reward Distribution': reward_distribution,
        'Mu': h1_loc,
        'Sigma': h1_scale,
        'Statistical Test': test_name,
        'Type I Error': type1_error,
        'Power': test_const,
        'Reward Std Dev': reward_std,
        'Minimum Effect Size': min_effect,
        'Family-Wise Error Rate Control': family_wise_error_control,
    }
    if max_reward is not None:
        user_inputs['Max Reward Per Step'] = max_reward

    if test_name == 't_control':
        user_inputs['Control Arm'] = t_control_param
        user_inputs['Tail Type'] = is_one_tail_control
    if test_name == 't_constant':
        user_inputs['Constant Threshold'] = t_constant_param
        user_inputs['Tail Type'] = is_one_tail_const
    if test_name == 'tukey':
        user_inputs['Tukey Test Type'] = tukey_test_type

    # --- Parse algorithm selection (per-algo config) ---
    algo_config_str = request.form.get('algo_config', '{}')
    algo_param_config = json.loads(algo_config_str)
    algo_names = list(algo_param_config.keys())
    sim_params['algo_names'] = algo_names
    sim_params['algo_param_config'] = algo_param_config
    algo_list = [ALGORITHM_MAP[name] for name in algo_names]
    use_adaptive = request.form.get('use_adaptive') == 'on'
    sim_params['use_adaptive'] = use_adaptive

    # --- Progress callbacks ---
    def on_progress(completed, total, algo_name, algo_param):
        sim_progress['completed_elapsed'] = time.time() - sim_progress['start_time']
        sim_progress.update({
            'running': completed < total,
            'completed': completed,
            'total': total,
            'algo_name': algo_name,
            'algo_param': algo_param,
            # Reset per-run fields when a new config starts
            'current_horizon': 0,
            'current_power': None,
            'current_phase': '',
            'batch_progress': 0,
        })

    def on_step(batch_idx, total_batches, t_actual, power):
        sim_progress['current_horizon'] = t_actual
        sim_progress['current_power'] = power
        sim_progress['current_phase'] = 'advancing' if power is None else 'checking'
        sim_progress['batch_progress'] = batch_idx / total_batches if total_batches > 0 else 0

    def on_result(result_row):
        completed_results.append(result_row)

    # --- Run simulation and get recommendation ---
    global cancel_requested
    cancel_requested = False
    completed_results.clear()
    sim_progress['running'] = True
    sim_progress['start_time'] = time.time()
    sim_progress['completed'] = 0
    sim_progress['completed_elapsed'] = 0
    sim_progress['max_horizon'] = horizon
    sim_progress['current_horizon'] = 0
    sim_progress['current_power'] = None
    sim_progress['current_phase'] = ''
    sim_progress['batch_progress'] = 0

    def check_cancel():
        return cancel_requested

    t_start = time.time()
    if use_adaptive:
        recommendations, plot_path, results_summary = get_recommendation_adaptive(
            n_arm=n_arm,
            horizon=horizon,
            n_rep=n_rep,
            reward_model=reward_model,
            h1_loc=h1_loc,
            h1_scale=h1_scale,
            reward_std=reward_std,
            test_procedure=test_procedure,
            algo_list=algo_list,
            progress_callback=on_progress,
            step_callback=on_step,
            algo_param_config=algo_param_config,
        )
    else:
        recommendations, plot_path, results_summary = get_recommendation(
            n_arm=n_arm,
            horizon=horizon,
            n_rep=n_rep,
            reward_model=reward_model,
            h1_loc=h1_loc,
            h1_scale=h1_scale,
            reward_std=reward_std,
            test_procedure=test_procedure,
            algo_list=algo_list,
            progress_callback=on_progress,
            result_callback=on_result,
            cancel_check=check_cancel,
            algo_param_config=algo_param_config,
        )
    sim_progress['running'] = False
    elapsed = round(time.time() - t_start, 1)

    return render_template(
        'recommend.html',
        user_inputs=user_inputs,
        recommendations=recommendations,
        plot_path=plot_path,
        results_summary=results_summary,
        chart_data_json=results_summary.get('chart_data_json', '[]'),
        sim_params=sim_params,
        elapsed_time=elapsed,
        max_reward=max_reward,
        power_constraint=results_summary.get('power_constraint', 0.80),
        n_h0_cores=results_summary.get('n_h0_cores'),
        n_h0_reps_per_core=results_summary.get('n_h0_reps_per_core'),
        available_algos=AVAILABLE_ALGOS,
    )


# /check_setting endpoint removed — GP interpolation no longer used in UI


@app.route('/run_additional', methods=['POST'])
def run_additional():
    """Run additional algorithm configurations and return results as JSON."""
    global cancel_requested
    data = request.get_json()

    algo_name = data['algo_name']
    params = [float(p) for p in data['params']]
    sp = data['sim_params']
    horizon = int(data.get('horizon') or sp['horizon'])
    n_rep = int(data.get('n_rep') or sp['n_rep'])
    n_h0_cores = data.get('n_h0_cores')
    n_h0_reps = data.get('n_h0_reps')

    # Build test procedure
    test_procedure = build_test_procedure(sp)

    # Apply H0 overrides (from this request, falling back to original sim_params)
    effective_cores = n_h0_cores or sp.get('n_h0_cores')
    effective_reps = n_h0_reps or sp.get('n_h0_reps')

    if effective_cores:
        if test_procedure.n_crit_approx_method == 'linear':
            test_procedure.n_crit_sim_groups = max(int(effective_cores) - 1, 1)
        else:
            test_procedure.n_crit_sim_groups = int(effective_cores)
    if effective_reps:
        test_procedure.n_crit_sim_rep = int(effective_reps)

    # Build sim config
    reward_model = REWARD_MODEL_MAP[sp['reward_distribution']]
    sim_config = create_simulation_config(
        n_arm=sp['n_arm'],
        horizon=horizon,
        n_rep=n_rep,
        reward_model=reward_model,
        arm_mean_reward_dist_loc=sp['h1_loc'],
        arm_mean_reward_dist_scale=sp['h1_scale'],
        reward_std=sp.get('reward_std'),
        test_procedure=test_procedure,
    )

    if algo_name not in ALGORITHM_MAP:
        return jsonify({'error': f'Unknown algorithm: {algo_name}'}), 400
    algo_class = ALGORITHM_MAP[algo_name]

    # Setup progress tracking
    cancel_requested = False
    completed_results.clear()
    sim_progress.update({
        'running': True,
        'start_time': time.time(),
        'completed': 0,
        'completed_elapsed': 0,
        'total': len(params),
        'max_horizon': horizon,
        'current_horizon': 0,
        'current_power': None,
        'current_phase': '',
        'batch_progress': 0,
    })

    results = []
    for i, param in enumerate(params):
        if cancel_requested:
            break
        sim_progress.update({
            'algo_name': algo_name,
            'algo_param': param,
        })

        result = run_single_check(sim_config, algo_class, param)
        results.append(result)
        sim_progress.update({
            'completed': i + 1,
            'completed_elapsed': time.time() - sim_progress['start_time'],
        })

    sim_progress['completed'] = len(results)
    sim_progress['completed_elapsed'] = time.time() - sim_progress['start_time']
    sim_progress['running'] = False

    return jsonify(results)


if __name__ == "__main__":
    print("Starting Flask app...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
