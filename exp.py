#OLD INDEX HTML (after line 26)
 <form action="/recommend" method="post">
        <div class="input-container2">
            <label for="n_arm">Number of Arms:</label>
            <input type="number" id="n_arm" name="n_arm" min="2" placeholder="Enter the number of arms in your project" required>
            <label for="horizon">Horizon:</label>            
            <input type="number" id="horizon" name="horizon" min="10" placeholder="Enter the max horizon (number of rounds) you would like to try" required>
            <label for="n_rep"> Number of Repetitions: </label>   
            <input type="number" id = "n_rep" name="n_rep" min="1" placeholder="Enter the number of simulation repetitions you would like" required>  
            <!-- <label for="n_opt_trials"> Number of Optimization Trials: </label>   
            <input type="number" id = "n_opt_trials" name="n_opt_trials" max="10" value=None placeholder="Enter the number of optimization trials you would like">   -->
        </div>

        <div class="input-container2">
            <label for="n_opt_trials"> Number of Optimization Trials: </label>   
            <input type="number" id = "n_opt_trials" name="n_opt_trials" min="1" max="10" value=None placeholder="Optional: Enter the number of optimization trials you would like">  
        </div>


        <div class = "input-container2">
            <label for="reward_distribution">Reward Distribution:</label>
            <select id="reward_distribution" name="reward_distribution" required onchange="handleTestSelection()">
                <option value="" selected disabled>Select your reward distribution</option>
                <option value="bernoulli">Bernoulli</option>
                <option value="gaussian">Gaussian</option>
            </select>
            <label for="h1_loc">Distribution Mean, &mu;:</label>
            <input type="number" id="h1_loc" name="h1_loc" min="0" step="0.01" value="0.5" required>
            <label for="h1_scale">Standard Deviation, &sigma;</label>
            <input type="number" id="h1_scale" name="h1_scale" min="0" step="0.01" value="0.15" required>
        </div>

        <!-- Additional inputs if gaussian reward model selected -->
        <div id="reward-inputs" class="conditional-inputs" style="display:none;">
            <label for="reward_std">Reward Standard Deviation:</label>
            <input type="number" id="reward_std" name="reward_std" placeholder="Std deviation of the reward distribution itself" style="width: 300px;">
        </div>

        <div class = "input-container3">
            <label for="test_name">Desired Statistical Test:</label>
            <select id="test_name" name="test_name" required onchange="handleTestSelection()">
                <option value="" selected disabled>Select the statistical test you would like to include</option>
                <option value="anova">ANOVA</option>
                <option value="t_control">T-Control</option>
                <option value="t_constant">T-Constant</option>
                <option value="tukey">Tukey</option>
            </select>
            <label for="type1_error_constraint">Type I Error Constraint:</label>
            <input type="number" id="type1_error_constraint" name="type1_error_constraint" placeholder="Enter Type I error constraint" min="0" max="1" step="0.001" value="0.05">
            <label for="step_cost">Step Cost:</label>
            <input type="number" id="step_cost" name="step_cost" min="0" max = "1.0" step="0.01"  value="0.05">
        </div>

        <div class = "input-container3">
            <label for="min_effect"> Minimum Effect Size (defines the smallest difference between arms to consider meaningful in power evaluation):</label>
            <input type="number" id="min_effect" name="min_effect" placeholder="Minimum detectable effect size for power calculations" min="0.01" max="1" step="0.001" value="0.1">
            <label for="family_wise_error_control"> Family-Wise Error Rate (FWER) Control:</label>
            <input type="checkbox" id="family_wise_error_control" name="family_wise_error_control" placeholder="Control FWER?" value="none">
        </div>

        <script>
            function handleTestSelection() {
                // Get the selected value from the dropdown
                const selectedTest = document.getElementById('test_name').value;
                const rewardDistribution = document.getElementById('reward_distribution').value;
        
                // Hide all conditional inputs by default
                document.querySelectorAll('.conditional-inputs').forEach(div => {
                    div.style.display = 'none';
                });
        
                // Show the relevant input fields based on the selected option
                if (selectedTest === 't_constant') {
                    document.getElementById('t-constant-inputs').style.display = 'block';
                } else if (selectedTest === 't_control') {
                    document.getElementById('t-control-inputs').style.display = 'block';
                }
                // Add more conditions for other options if needed
                if (rewardDistribution === 'gaussian') {
                    document.getElementById('reward-inputs').style.display = 'block';
                }
            }
        </script>

        <!--sync slider bar with float input for power -->        
        <div class="input-container">
            <label for="test_const">Desired Power:</label>
            <!-- <span id="powerValue">0.8</span> -->
            <input type="range"  id="powerSlider" name="powerSlider" min="0" max="1" step="0.01" value="0.8" oninput="updateValue(this.value)" required>
            <input type="number" id="test_const" name ="test_const" min="0" max="1" step="0.01" value="0.8" oninput="updateValue(this.value)" required>
        </div>

        <script>
            function updateValue(value) {
                // document.getElementById('powerValue').textContent = value;
                document.getElementById('test_const').value = value;
                document.getElementById('powerSlider').value = value;
            }
        </script>
        
     <!-- More Additional inputs for specific options -->
     <div id="reward-inputs" class="conditional-inputs" style="display:none;">
        <label for="reward_std">Reward Standard Deviation:</label>
        <input type="number" id="reward_std" name="reward_std" placeholder="Std deviation of the reward distribution itself" style="width: 300px;">
    </div>

     <div id="t-control-inputs" class="conditional-inputs" style="display:none;">
        <label for="t_control_param">Control Arm:</label>
        <input type="number" id="t_control_param" name="t_control_param" placeholder="Default arm is (index) 0" style="width: 300px;">
        <!-- <label for="min_effect">Minimum Effect Size:</label> -->
        <!-- <input type="number" id="min_effect" name="min_effect"  min="0" step="0.01" style="width: 300px;"> -->
        <select id="is_one_tail_control" name="is_one_tail_control" style="width: 300px;" required>
            <option value="" selected disabled>One- or Two-sided test?</option>
            <option value="one-sided">One-sided</option>
            <option value="two-sided">Two-sided</option>
        </select>
    </div>
        
    <div id="t-constant-inputs" class="conditional-inputs" style="display:none;">
        <label for="t_constant_param">Constant Value:</label>
        <input type="number" id="t_constant_param" name="t_constant_param" min="0" max="1" step="0.01" style="width: 300px;">
        <!-- <label for="min_effect2">Minimum Effect Size:</label> -->
        <!-- <input type="number" id="min_effect2" name="min_effect2"  min="0" step="0.01" style="width: 300px;" > -->
        <select id="is_one_tail_const" name="is_one_tail_const" style="width: 300px;">
            <option value="" selected disabled>One- or Two-sided test?</option>
            <option value="one-sided">One-sided</option>
            <option value="two-sided">Two-sided</option>
        </select>
    </div>

    <div id="tukey-inputs" class="conditional-inputs" style="display:none;">
        <select id="tukey_test_type" name="tukey_test_type" style="width: 300px;"required>
            <option value="" selected disabled>Select Test type</option>
            <option value="all-pair-wise">All-Pairwise</option>
            <option value="distinct-best-arm">Distinct-Best Arm</option>
        </select>
    </div>

    <div class = "input-container4">
        <label for="algo">Algorithm:</label>
        <select id="algo" name="algo[]" multiple required>
            <option value="" selected disabled>Select one or more algorithms</option>
            <option value="TSProbClip">TSProbClip</option>
            <option value="EpsTS">EpsTS</option>
            <option value="TSTopUR">TSTopUR</option>
        </select>

        <label for="granularity">Granularity:</label>
        <input type="number" id="granularity" name="granularity" min="5" max="30" value="21">
    </div>

        <button type="submit">Get Recommendation</button>
    </form>
</br>
</body>
</html>

#old recommend.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAB Recommender</title>
    <link href="{{url_for('static', filename='styles/style2.css')}}" rel="stylesheet" />
</head>
<body>
    <h1>Here are our recommendations for your project!</h1> 
    <!-- <p> Here are our recommendations for your project!</p> -->


    {% if user_inputs %}
    <div class="user-input-list">
        <h4>Your Inputs:</h4>
        <ul>
            {% for label, value in user_inputs.items() %}
                <li><strong>{{ label }}:</strong> {{ value }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    <div class="image-container">
        <h2>Performance Curves</h2>
        <img src="{{ plot_path }}" alt="Performance Curves" > 
    </div>
   
    <!-- width="300" height="200" -->

    {% if recommendations %}
    <h3>Recommended for you:</h3>
    <ul class="recommendation-list">
        {% for item in recommendations %}
            <li>{{ item }}</li>
        {% endfor %}
    </ul>
    {% endif %}

    <!-- <form action="/recommend"> 
        <input type="text" id="n_rep" name="n_rep" placeholder="Enter arm names separated by commas and a space" required>
        <label for="n_rep">Number of Reps:</label>
        <input type="number" id="n_arms" name="n_arms" min="2" required>
        <label for="n_arms">Number of Arms:</label>
        <input type="number" id="horizon" name="horizon" min="1" required>
        <label for="horizon">Horizon (number of rounds):</label>
        <button type="submit">Get Recommendation</button>
    </form>
</body>
</html> -->