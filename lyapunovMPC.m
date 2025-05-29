function [time, inputs, outputs, references, current_mode] = lyapunovMPC(modes, ...
    mode_switch_time, sim_time, input_range, delta_u_max, noise_std, setpoint, ...
    alpha, beta, T_p, T_c, J_set, max_iter)
% LYAPUNOVMPC Runs a simulation with Lyapunov-based Model Predictive Control
%
% Inputs:
%   modes           - Array of system modes
%   mode_switch_time- Time steps to switch modes
%   sim_time        - Total simulation time
%   input_range     - Range of control inputs [min, max]
%   delta_u_max     - Maximum input rate
%   noise_std       - Standard deviation of Gaussian noise
%   setpoint        - Control setpoint
%   alpha           - Weight for tracking error
%   beta            - Weight for control effort
%   T_p             - Prediction horizon
%   T_c             - Control horizon
%   J_set           - Optimization termination threshold
%   max_iter        - Maximum optimization iterations
%
% Outputs:
%   time            - Time vector
%   inputs          - Control inputs
%   outputs         - System outputs
%   references      - Reference setpoints
%   current_mode    - Current system mode index at each time step

% Initialize arrays
time = 1:sim_time;
inputs = zeros(sim_time, 1);
outputs = zeros(sim_time, 1);
references = setpoint * ones(sim_time, 1);
current_mode = ones(sim_time, 1);

% Extract constraints
u_min = input_range(1);
u_max = input_range(2);

% Initialize system with zeros
inputs(1:2) = 0.5 * (u_min + u_max); % Start with middle point
outputs(1:2) = 0;

% Apply the initial mode
for i = 2:mode_switch_time(1)
    current_mode(i) = 1;
end

% Apply mode switches
for m = 2:length(mode_switch_time)
    start_idx = mode_switch_time(m-1);
    if m == length(mode_switch_time)
        end_idx = sim_time;
    else
        end_idx = mode_switch_time(m) - 1;
    end
    current_mode(start_idx:end_idx) = m-1;
end

% Train a DNN model for prediction (same as DNN-MPC)
fprintf('Training DNN model for Lyapunov-MPC...\n');
dnn_model = trainDNNModel(modes, input_range, noise_std);

% Lyapunov function parameters
lyap_p = 1.0;      % Positive definite parameter for V(e) = p * e^2
lyap_epsilon = 1e-3; % Small positive value for stability constraint

% Main simulation loop
for t = 3:sim_time
    % Get current mode
    mode_idx = current_mode(t);
    
    % Create input sequence for LMPC
    input_sequence = [inputs(t-1), inputs(t-2), outputs(t-1), outputs(t-2)];
    
    % Current error
    current_error = outputs(t-1) - setpoint;
    
    % Run Lyapunov-MPC optimization to get optimal control input
    u_optimal = lyapunovMPCOptimization(dnn_model, input_sequence, setpoint, current_error, ...
        alpha, beta, T_p, T_c, u_min, u_max, delta_u_max, J_set, max_iter, ...
        lyap_p, lyap_epsilon);
    
    % Apply optimal control input
    inputs(t) = u_optimal;
    
    % Simulate system response
    outputs(t) = hammersteinSystem(inputs(t-1), inputs(t-2), outputs(t-1), outputs(t-2), ...
        modes(mode_idx), noise_std);
end

end

function dnn_model = trainDNNModel(modes, input_range, noise_std)
% Train a DNN for prediction in Lyapunov-MPC
% This is the same function as in dnnMPC.m
    
% Data collection parameters
num_modes = length(modes);
data_points_per_mode = 1000;
total_points = num_modes * data_points_per_mode;
    
% Generate training data
X_data = zeros(total_points, 4);  % [u(t), u(t-1), y(t), y(t-1)]
Y_data = zeros(total_points, 1);  % y(t+1)
    
% Generate random inputs for training
u_min = input_range(1);
u_max = input_range(2);
    
data_idx = 1;
for mode_idx = 1:num_modes
    current_mode = modes(mode_idx);
        
    % Initialize for this mode with padding for proper indexing
    u_seq = u_min + (u_max - u_min) * rand(data_points_per_mode + 3, 1);
    y_seq = zeros(data_points_per_mode + 3, 1);
        
    % Initial conditions
    y_seq(1:3) = 0;
        
    % Generate sequence for this mode
    for i = 4:length(y_seq)
        y_seq(i) = hammersteinSystem(u_seq(i-1), u_seq(i-2), y_seq(i-1), y_seq(i-2), ...
            current_mode, noise_std);
    end
        
    % Store inputs and outputs
    for i = 1:data_points_per_mode
        X_data(data_idx, :) = [u_seq(i+2), u_seq(i+1), y_seq(i+2), y_seq(i+1)];
        Y_data(data_idx, :) = y_seq(i+3);
        data_idx = data_idx + 1;
    end
end
    
% Shuffle data
shuffle_idx = randperm(total_points);
X_data = X_data(shuffle_idx, :);
Y_data = Y_data(shuffle_idx, :);
    
% Split into training (80%) and validation (20%) sets
train_ratio = 0.8;
train_size = round(total_points * train_ratio);
    
X_train = X_data(1:train_size, :);
Y_train = Y_data(1:train_size, :);
X_val = X_data(train_size+1:end, :);
Y_val = Y_data(train_size+1:end, :);
    
% Create a feedforward neural network
input_size = size(X_train, 2);
hidden_layers = [24, 16]; % Two hidden layers with 24 and 16 neurons
output_size = 1;
    
% Create and train the DNN
dnn_model = feedforwardnet(hidden_layers);
dnn_model.trainParam.epochs = 200;
dnn_model.trainParam.min_grad = 1e-5;
dnn_model.trainParam.showWindow = true;
dnn_model.divideParam.trainRatio = 0.8;
dnn_model.divideParam.valRatio = 0.2;
dnn_model.divideParam.testRatio = 0;
    
% Train the network
[dnn_model, ~] = train(dnn_model, X_train', Y_train');
    
% Evaluate model performance
Y_pred = dnn_model(X_val')';
val_mse = mean((Y_pred - Y_val).^2);
fprintf('DNN Model Validation MSE: %.6f\n', val_mse);
end

function u_optimal = lyapunovMPCOptimization(dnn_model, input_sequence, setpoint, current_error, ...
    alpha, beta, T_p, T_c, u_min, u_max, delta_u_max, J_set, max_iter, ...
    lyap_p, lyap_epsilon)
% Lyapunov-MPC optimization with stability constraints
    
% Current input and state values
u_current = input_sequence(1);
u_prev = input_sequence(2);
y_current = input_sequence(3);
y_prev = input_sequence(4);
    
% Create reference trajectory (constant setpoint)
R = setpoint * ones(T_p, 1);
    
% Initialize control sequence
U_k = u_current * ones(T_c, 1);
delta_U_k = zeros(T_c, 1);
    
% Current Lyapunov function value
V_current = lyap_p * current_error^2;
    
% Iterative optimization
iter = 0;
J_val = Inf;
learning_rate = 0.01;
    
% Flag to track if Lyapunov constraint is violated
lyap_violated = false;
    
while (J_val > J_set) && (iter < max_iter)
    % Predict system outputs using DNN
    Y_pred = predictMultiStepDNN(dnn_model, input_sequence, U_k);
        
    % Calculate objective function value
    tracking_error = R - Y_pred;
    J_val = tracking_error' * alpha * tracking_error + delta_U_k' * beta * delta_U_k;
        
    % Check Lyapunov stability constraint for the first prediction step
    predicted_error = Y_pred(1) - setpoint;
    V_next = lyap_p * predicted_error^2;
    lyap_constraint = V_next - (V_current - lyap_epsilon);
        
    % Calculate Jacobian using numerical differentiation
    delta = 1e-4;
    jacobian = zeros(T_p, T_c);
        
    for j = 1:T_c
        % Create perturbed control sequence
        U_perturb = U_k;
        U_perturb(j) = U_perturb(j) + delta;
            
        % Predict with perturbation
        Y_perturb = predictMultiStepDNN(dnn_model, input_sequence, U_perturb);
            
        % Calculate partial derivatives
        jacobian(:, j) = (Y_perturb - Y_pred) / delta;
    end
        
    % Calculate gradient of objective function
    grad_obj = -alpha * jacobian' * tracking_error + beta * delta_U_k;
        
    % If Lyapunov constraint is violated, add gradient component
    if lyap_constraint > 0
        lyap_violated = true;
            
        % Calculate gradient of Lyapunov constraint
        grad_lyap = zeros(T_c, 1);
            
        for j = 1:T_c
            % Create perturbed control sequence
            U_perturb = U_k;
            U_perturb(j) = U_perturb(j) + delta;
                
            % Predict with perturbation for first step only
            Y_first_perturb = predictMultiStepDNN(dnn_model, input_sequence, U_perturb);
            predicted_error_perturb = Y_first_perturb(1) - setpoint;
            V_next_perturb = lyap_p * predicted_error_perturb^2;
            lyap_constraint_perturb = V_next_perturb - (V_current - lyap_epsilon);
                
            % Gradient of Lyapunov constraint
            grad_lyap(j) = (lyap_constraint_perturb - lyap_constraint) / delta;
        end
            
        % Combine gradients with weight on Lyapunov constraint
        gradient = grad_obj + 10 * grad_lyap;  % Higher weight on stability
    else
        gradient = grad_obj;
    end
        
    % Update step
    delta_U_new = delta_U_k - learning_rate * gradient;
        
    % Apply constraints on control increments
    delta_U_k = min(delta_u_max, max(-delta_u_max, delta_U_new));
        
    % Calculate new control inputs
    U_k_new = u_current + delta_U_k;
        
    % Apply constraints on control inputs
    U_k = min(u_max, max(u_min, U_k_new));
        
    % Recalculate increments
    delta_U_k = U_k - u_current;
        
    % Adjust learning rate (simple decay)
    learning_rate = learning_rate * 0.99;
        
    iter = iter + 1;
end
    
if lyap_violated
    fprintf('Lyapunov constraint violated. Stability not guaranteed.\n');
end
    
% Return optimal control input (first element)
u_optimal = U_k(1);
end

function Y_pred = predictMultiStepDNN(dnn_model, input_sequence, U_future)
% Predict multiple steps ahead using the DNN model
% This is the same function as in dnnMPC.m
    
% Extract current values
u_current = input_sequence(1);
u_prev = input_sequence(2);
y_current = input_sequence(3);
y_prev = input_sequence(4);
    
% Number of steps to predict
prediction_horizon = 2;
Y_pred = zeros(prediction_horizon, 1);
    
% Initial state
u_next = U_future(1);  % First future control input
    
% Predict first step
x_new = [u_next, u_current, y_current, y_prev]';
y_next = dnn_model(x_new);
Y_pred(1) = y_next;
    
% Predict remaining steps if needed
if prediction_horizon > 1
    for i = 2:prediction_horizon
        % Update inputs for next prediction
        u_idx = min(i, length(U_future));
        u_new_next = U_future(u_idx);
            
        % Update sequence for prediction
        x_new = [u_new_next, u_next, y_next, y_current]';
            
        % Predict next output
        y_new_next = dnn_model(x_new);
        Y_pred(i) = y_new_next;
            
        % Update for next iteration
        u_prev = u_next;
        u_next = u_new_next;
        y_prev = y_current;
        y_current = y_next;
        y_next = y_new_next;
    end
end
end 