import numpy as np
from flask import Flask, render_template, request
from waitress import serve

from bandit_simulation import ANOVA, EpsTS, TConstant, TControl, TSProbClip, TSTopUR, Tukey
from recommendation import get_recommendation

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=["post"])
def recommend():
    print("Form submitted!")

    # --- Parse core simulation parameters ---
    n_arm = int(request.form['n_arm'])
    horizon = int(request.form['horizon'])
    n_rep = int(request.form['n_rep'])

    reward_distribution = request.form['reward_distribution']
    reward_model_mapping = {
        'bernoulli': np.random.binomial,
        'gaussian': np.random.normal
    }
    reward_model = reward_model_mapping[reward_distribution]
    reward_std = float(request.form['reward_std']) if reward_distribution == 'gaussian' else None

    h1_loc = float(request.form['h1_loc'])
    h1_scale = float(request.form['h1_scale'])

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

    # --- Create test procedure ---
    test_procedure: ANOVA | TControl | TConstant | Tukey
    if test_name == 'anova':
        test_procedure = ANOVA(
            type1_error_constraint=type1_error,
            power_constraint=test_const,
            min_effect=min_effect,
            family_wise_error_control=family_wise_error_control
        )
    elif test_name == 't_control':
        test_type = 'one-sided' if is_one_tail_control == 'one-sided' else 'two-sided'
        test_procedure = TControl(
            type1_error_constraint=type1_error,
            power_constraint=test_const,
            min_effect=min_effect,
            family_wise_error_control=family_wise_error_control,
            test_type=test_type,
            control_group_index=t_control_param
        )
    elif test_name == 't_constant':
        test_type = 'one-sided' if is_one_tail_const == 'one-sided' else 'two-sided'
        test_procedure = TConstant(
            type1_error_constraint=type1_error,
            power_constraint=test_const,
            min_effect=min_effect,
            family_wise_error_control=family_wise_error_control,
            test_type=test_type,
            constant_threshold=t_constant_param
        )
    else:  # tukey
        tukey_type = tukey_test_type if tukey_test_type in ('all-pair-wise', 'distinct-best-arm') \
            else 'distinct-best-arm'
        test_procedure = Tukey(
            type1_error_constraint=type1_error,
            power_constraint=test_const,
            min_effect=min_effect,
            family_wise_error_control=family_wise_error_control,
            test_type=tukey_type,
        )

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

    if test_name == 't_control':
        user_inputs['Control Arm'] = t_control_param
        user_inputs['Tail Type'] = is_one_tail_control
    if test_name == 't_constant':
        user_inputs['Constant Threshold'] = t_constant_param
        user_inputs['Tail Type'] = is_one_tail_const
    if test_name == 'tukey':
        user_inputs['Tukey Test Type'] = tukey_test_type

    # --- Parse algorithm selection ---
    algo_names = request.form.getlist('algo[]')
    algorithm_mapping = {
        "TSProbClip": TSProbClip,
        "EpsTS": EpsTS,
        "TSTopUR": TSTopUR,
    }
    algo_list = [algorithm_mapping[name] for name in algo_names]
    granularity = int(request.form.get('granularity', 21))

    # --- Run simulation and get recommendation ---
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
        granularity=granularity,
    )

    return render_template(
        'recommend.html',
        user_inputs=user_inputs,
        recommendations=recommendations,
        plot_path=plot_path,
        results_summary=results_summary,
        chart_data_json=results_summary.get('chart_data_json', '[]'),
    )


if __name__ == "__main__":
    print("Starting Flask app...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
