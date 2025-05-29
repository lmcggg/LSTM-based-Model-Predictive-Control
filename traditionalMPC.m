function [time, inputs, outputs, references, current_mode] = traditionalMPC(modes, ...
    mode_switch_time, sim_time, input_range, delta_u_max, noise_std, setpoint, ...
    alpha, beta, T_p, T_c, J_set, max_iter)
% TRADITIONALMPC Runs a simulation with traditional Model Predictive Control
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

% Hammerstein model parameters for linearization
% These are the matrices for a linear state-space model approximation 
% of the Hammerstein system: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k)
A = [0.6, -0.1; 
     1.0,  0.0];
B = [1.2; 0.0];
C = [1.0, -0.1];

% Main simulation loop
for t = 3:sim_time
    % Get current mode
    mode_idx = current_mode(t);
    current_mode_params = modes(mode_idx);
    
    % Current state estimation (from past inputs/outputs)
    x_current = [outputs(t-1); outputs(t-2)];
    
    % Adapt model for current operating mode
    % Scale B matrix according to the current mode's parameters
    % This is a simple adaptation strategy for the multi-mode system
    B_scaled = B * (current_mode_params.a_m / 1.0); % Normalize relative to mode 2
    
    % Run traditional MPC optimization
    u_optimal = traditionalMPCOptimization(A, B_scaled, C, x_current, setpoint, ...
        alpha, beta, T_p, T_c, u_min, u_max, delta_u_max, J_set, max_iter, inputs(t-1));
    
    % Apply optimal control input
    inputs(t) = u_optimal;
    
    % Simulate system response with the actual nonlinear Hammerstein system
    outputs(t) = hammersteinSystem(inputs(t-1), inputs(t-2), outputs(t-1), outputs(t-2), ...
        modes(mode_idx), noise_std);
end
end

function u_optimal = traditionalMPCOptimization(A, B, C, x_current, setpoint, alpha, beta, ...
    T_p, T_c, u_min, u_max, delta_u_max, J_set, max_iter, u_prev)
% Traditional MPC optimization using linear prediction model
    
% Formulate the prediction matrices
[F, G] = getPredictionMatrices(A, B, C, T_p, T_c);
    
% Reference trajectory
R = setpoint * ones(T_p, 1);
    
% Initialize control sequence with previous control
delta_U = zeros(T_c, 1);
U = u_prev * ones(T_c, 1);
    
% Free response (if no control action is taken)
f = F * x_current;
    
% Iterative optimization using projected gradient descent
J_val = Inf;
iter = 0;
learning_rate = 0.01;
    
while (J_val > J_set) && (iter < max_iter)
    % Predicted outputs
    Y_pred = f + G * delta_U;
        
    % Calculate cost function
    tracking_error = R - Y_pred;
    J_val = tracking_error' * alpha * tracking_error + delta_U' * beta * delta_U;
        
    % Gradient of cost function
    gradient = -alpha * G' * tracking_error + beta * delta_U;
        
    % Update step
    delta_U_new = delta_U - learning_rate * gradient;
        
    % Apply constraints on control increments
    delta_U_new = min(delta_u_max, max(-delta_u_max, delta_U_new));
        
    % Calculate new control inputs
    U_new = U + delta_U_new;
        
    % Apply constraints on control inputs
    U_new = min(u_max, max(u_min, U_new));
        
    % Recalculate increments based on constrained inputs
    delta_U = U_new - U;
    U = U_new;
        
    iter = iter + 1;
end
    
% Return optimal control input (first element)
u_optimal = U(1);
end

function [F, G] = getPredictionMatrices(A, B, C, T_p, T_c)
% Calculate prediction matrices for the traditional MPC
nx = size(A, 1);  % Number of states
nu = size(B, 2);  % Number of inputs
    
% Initialize matrices
F = zeros(T_p, nx);
G = zeros(T_p, T_c);
    
% Calculate F matrix (free response)
for i = 1:T_p
    F(i, :) = C * (A^i);
end
    
% Calculate G matrix (forced response)
for i = 1:T_p
    for j = 1:min(i, T_c)
        G(i, j) = C * (A^(i-j)) * B;
    end
end
end 