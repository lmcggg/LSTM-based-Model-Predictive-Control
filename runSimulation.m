function [time, inputs, outputs, references, current_mode] = runSimulation(lstm_model, modes, ...
    mode_switch_time, sim_time, input_range, delta_u_max, noise_std, setpoint, alpha, beta, ...
    T_p, T_c, J_set, max_iter, learning_rate_0, decay_rate, time_step)
% RUNSIMULATION Runs a simulation of the LSTM-MPC control system
%
% Inputs:
%   lstm_model      - Trained LSTM model
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
%   learning_rate_0 - Initial learning rate
%   decay_rate      - Learning rate decay rate
%   time_step       - Time step for input sequence
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

% Main simulation loop
for t = 3:sim_time
    % Get current mode
    mode_idx = current_mode(t);
    
    % Create input sequence for LSTM-MPC
    input_sequence = [inputs(t-1), inputs(t-2), outputs(t-1), outputs(t-2)];
    
    % Run MPC optimization to get optimal control input
    u_optimal = mpcOptimization(lstm_model, input_sequence, setpoint, alpha, beta, T_p, T_c, ...
        u_min, u_max, delta_u_max, J_set, max_iter, learning_rate_0, decay_rate);
    
    % Apply optimal control input
    inputs(t) = u_optimal;
    
    % Simulate system response
    outputs(t) = hammersteinSystem(inputs(t-1), inputs(t-2), outputs(t-1), outputs(t-2), ...
        modes(mode_idx), noise_std);
end

end 