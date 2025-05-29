function u_optimal = mpcOptimization(lstm_model, input_sequence, setpoint, alpha, beta, T_p, T_c, ...
    u_min, u_max, delta_u_max, J_set, max_iter, learning_rate_0, decay_rate)
% MPCOPTIMIZATION Performs MPC optimization using adaptive projected gradient descent
%
% Inputs:
%   lstm_model     - Trained LSTM model
%   input_sequence - Current input sequence [u(t), u(t-1), y(t), y(t-1)]
%   setpoint       - Reference setpoint value
%   alpha          - Weight for tracking error
%   beta           - Weight for control effort
%   T_p            - Prediction horizon
%   T_c            - Control horizon
%   u_min          - Minimum input constraint
%   u_max          - Maximum input constraint
%   delta_u_max    - Maximum input rate constraint
%   J_set          - Optimization termination threshold
%   max_iter       - Maximum optimization iterations
%   learning_rate_0- Initial learning rate
%   decay_rate     - Learning rate decay rate
%
% Output:
%   u_optimal      - Optimal control action

% Extract current input
u_current = input_sequence(1);

% Create reference trajectory (constant setpoint)
R = setpoint * ones(T_p, 1);

% Initialize control sequence (use previous control as initial guess)
U_k = u_current * ones(T_c, 1);
delta_U_k = zeros(T_c, 1);

% Small perturbation for numerical differentiation
delta = 1e-4;

% Initialize learning rate
learning_rate = learning_rate_0;

% Iterative optimization
iter = 0;
J_val = Inf;

while (J_val > J_set) && (iter < max_iter)
    % Step 1: Predict system outputs using current control sequence
    Y_pred = predictMultiStep(lstm_model, input_sequence, delta_U_k);
    
    % Step 2: Calculate Jacobian matrix
    jacobian = calculateJacobian(lstm_model, input_sequence, T_c, delta);
    
    % Step 3: Calculate objective function value
    tracking_error = R - Y_pred;
    J_val = tracking_error' * alpha * tracking_error + delta_U_k' * beta * delta_U_k;
    
    % Step 4: Calculate gradient (using Eq. 14 from the paper)
    gradient = -alpha * jacobian' * tracking_error + beta * delta_U_k;
    
    % Step 5: Calculate update step (Eq. 15 from the paper)
    update_factor = learning_rate / (1 + learning_rate * beta);
    delta_U_step = update_factor * (alpha * jacobian' * tracking_error);
    
    % Step 6: Apply projection for increments (Eq. 19)
    delta_U_projected = min(delta_u_max, max(-delta_u_max, delta_U_step));
    
    % Step 7: Update control sequence and apply constraints (Eq. 16 & 18)
    delta_U_k = delta_U_projected;
    U_k_new = U_k + delta_U_k;
    U_k = min(u_max, max(u_min, U_k_new));
    
    % Step 8: Update learning rate with decay (Eq. 20)
    learning_rate = learning_rate * exp(-decay_rate * iter);
    
    % Increment iteration counter
    iter = iter + 1;
end

% Return optimal control input (first element of sequence)
u_optimal = U_k(1);

end 