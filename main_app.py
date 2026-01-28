from flask import Flask, render_template, request, redirect, url_for
from waitress import serve #to help serve web app into production
from recommendation import get_recommendation #define this in my recommendation.py file
import numpy as np
import pandas as pd
from bandit_algorithm import TSProbClip, EpsTS, TSTopUR #TODO: figure out how to import from this


app = Flask(__name__)

#other options would be just to have a single route for the home page, or should the home page be some type of explanation about the project
#and then the recommendation page would be a separate route?
@app.route('/') #home page
@app.route('/index') #purpose of this route is to allow the user to access the home page using either '/' or '/index'
def index():
    return render_template('index.html')

@app.route('/recommend', methods=["post"]) #process user input amd return recommendation
def recommend():
    # print(request.form)
    print ("Form submitted!") #shows in terminal/console
    #get the user input from the form
    n_arm = int(request.form['n_arm']) 
    horizon = int(request.form['horizon'])
    n_rep = int(request.form['n_rep'])
    n_opt_trials = int(request.form['n_opt_trials']) 
    reward_distribution = request.form['reward_distribution']
    h1_loc = float(request.form['h1_loc'])
    h1_scale = float(request.form['h1_scale'])
    test_name = request.form['test_name'] 
    type1_error = float(request.form['type1_error_constraint'])
    test_const = float(request.form['test_const'])
    reward_std = float(request.form['reward_std']) if reward_distribution == 'gaussian' else None
    step_cost = float(request.form['step_cost']) 
    min_effect = float(request.form['min_effect']) 
    family_wise_error_control = request.form.get('family_wise_error_control') == 'on'  #checkbox input
    print("FWER Control:", family_wise_error_control)  # Debugging output
    
    #TODO: ensure that these conditional inputs are still applicable from test_procedure_config
    # Optional/conditional inputs - giving default values if not provided to avoid server errors
    reward_std = float(request.form.get('reward_std', None)) #GP - isn't this redundant with above?
    t_control_param = int(request.form.get('t_control_param', 0)) 
    t_constant_param = float(request.form.get('t_constant_param', 0.0))
    # min_effect = float(request.form.get('min_effect', 0.0)) if request.form.get('min_effect') else None
    # min_effect2 = float(request.form.get('min_effect2', 0.0)) if request.form.get('min_effect2') else None
    is_one_tail_control = request.form.get('is_one_tail_control', None)
    is_one_tail_const = request.form.get('is_one_tail_const', None)
    tukey_test_type = request.form.get('tukey_test_type', None) #GP - relevance? do I need to do these for other test types?

    
    #store the user input in a dictionary to pass to the template
    user_inputs = {
        'Number of Arms': n_arm, 
        'Horizon': horizon,
        'Number of Repetitions': n_rep,
        'Number of Optimization Trials': n_opt_trials,
        'Reward Distribution': reward_distribution,
        'Mu': h1_loc,
        'Sigma': h1_scale,
        'Statistical Test': test_name,
        'Type I Error': type1_error,
        'Power': test_const,
        'Reward Std Dev': reward_std,
        'Step Cost': step_cost,
        'Minimum Effect Size': min_effect,
        'Family-Wise Error Rate Control': family_wise_error_control
        } 
    
    # Add conditionals if present
    if reward_std is not None:
        user_inputs['Reward Standard Deviation'] = reward_std
    if t_control_param is not None:
        user_inputs['Control Arm'] = t_control_param
    if t_constant_param is not None:
        user_inputs['Constant'] = t_constant_param
    # if min_effect is not None:
    #     user_inputs['Minimum Effect Size (Control)'] = min_effect
    # if min_effect2 is not None:
    #     user_inputs['Minimum Effect Size (Constant)'] = min_effect2
    if is_one_tail_control is not None:
        user_inputs['Tail Type (Control)'] = is_one_tail_control
    if is_one_tail_const is not None:
        user_inputs['Tail Type (Constant)'] = is_one_tail_const
    if tukey_test_type is not None:
        user_inputs['Tukey Test Type'] = tukey_test_type
    
    # get sweep specifications from user input
    #TODO: add to user input form to allow user to select algorithm and level of granularity
    algo_names = request.form.getlist('algo[]')  #get the selected algorithms from the form - allow user to choose from list of algorithms
    
    # Mapping from string names to Python objects
    algorithm_mapping = {
    "TSProbClip": TSProbClip,
    "EpsTS": EpsTS,
    "TSTopUR": TSTopUR }
    
    algo = [algorithm_mapping[name] for name in algo_names]
    print("Selected Algorithms:", algo)  # Debugging output
    
    granularity = int(request.form.get('granularity', 21)) #default to 21 if not provided - allow user to specify how granular they want the parameter sweep to be
    sweeps = [
        {"algo": algo}, #ensure this matches the name attribute in the select element in index.html - ["EpsTS"]}
        {"algo_param_list": list(map(float, np.linspace(0.0, 1.0, granularity)))} #TODO: allow the user to specify how many values they want - how granular - replace 21
    ]

    #call the get_recommendation function to handle both simulations and plotting results
    #TODO: edit the get_recommendation function (both in recommendation.py and enter all the input parameters in this function call below) to take in all these parameters and use them in the logic to recommend an algorithm
    #last - recommendations, plot_path = get_recommendation(n_arm, horizon) #call the function to get the recommendation based on the number of arms and horizon
    # recommendations, plot_path = get_recommendation(n_arm=n_arm, horizon=horizon, n_rep=n_rep, reward_distribution=reward_distribution, h1_loc=h1_loc, h1_scale=h1_scale, test_name=test_name, type1_error=type1_error, test_const=test_const, sweeps=sweeps, selectors=None)
    recommendations = get_recommendation(n_arm, horizon, algo) #call the function to get the recommendation based on all user inputs
    return render_template('recommend.html', user_inputs=user_inputs, recommendations=recommendations)
    # return render_template('recommend.html', user_inputs=user_inputs, recommendations=recommendations, plot_path=plot_path)
   
    # parameters = {"epsilon": 0.1} #placeholder for the parameters of the algorithm
    # # return render_template('recommend.html',
    #                        algorithm=algorithm,
    #                        parameters=parameters)

#dynamically update the plot based on new selectors provided by user
# @app.route('/update_plot', methods=["POST"])
# def update_plot():
#     # Collect selectors from the form
#     #TODO: if a single algorithm is selected, allow w and param modes but if multiple algorithms are selected, only allow w mode
#     selectors = []
#     for i in range(1, 5):  # Assuming up to 4 selectors for simplicity
#         algo_name = request.form.get(f"algo_name_{i}")
#         mode = request.form.get(f"mode_{i}")
#         value = request.form.get(f"value_{i}")
#         if algo_name and mode and value:
#             selectors.append((algo_name, mode, float(value)))

#     # Load the precomputed simulation results
#     df = pd.read_csv("simulation_results.csv")

#     # Generate updated plot
#     from main import select_curves_relative, plot_curves
#     curves = select_curves_relative(df, selectors, w_values=range(1, 16))
#     plot_path = "static/plot.png"
#     plot_curves(curves, output_path=plot_path)

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8000, debug=True) #run the Flask app on port 8000, accessible from any IP address
    serve(app, host='0.0.0.0', port=8000, threads=4) #use waitress to serve the app, allowing for multiple threads to handle requests

# to test locally and view in browser, run this file and go to http://localhost:8000