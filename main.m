%% LSTM-MPC: A Deep Learning Based Predictive Control Method for Multimode Process Control
% Main simulation script

close all; clear; clc;

%% Define system parameters
% Hammerstein system parameters
modes = [
    % Mode 1
    struct('a_m', 1.5, 'b_m', 1.5), 
    % Mode 2
    struct('a_m', 1.0, 'b_m', 1.0), 
    % Mode 3
    struct('a_m', 0.5, 'b_m', 0.5)
];

% Simulation parameters
sim_time = 300;         % Total simulation time steps
mode_switch_time = [1, 100, 200]; % Time steps to switch modes
noise_std = 0.01;       % Standard deviation of Gaussian noise
input_range = [0, 1];   % Input range constraints
setpoint = 0.6;         % Control setpoint

% LSTM model parameters
lstm_hidden_size = 16;  % Number of hidden nodes
time_step = 2;          % Sequence length for LSTM
input_dim = 4;          % [u(t), u(t-1), y(t), y(t-1)]
output_dim = 1;         % y(t+1)

% MPC parameters
alpha = 1;              % Weight for tracking error
beta = 1;               % Weight for control effort
T_p = 2;                % Prediction horizon
T_c = 1;                % Control horizon
delta_u_max = 0.1;      % Maximum input rate
J_set = 1e-4;           % Optimization termination threshold
max_iter = 50;          % Maximum optimization iterations
learning_rate_0 = 0.01; % Initial learning rate
decay_rate = 0.05;      % Learning rate decay

%% Data collection for LSTM training
fprintf('Collecting training data...\n');
[train_data, test_data] = collectTrainingData(modes, input_range, noise_std);

%% Train LSTM model
fprintf('Training LSTM model...\n');
lstm_model = trainLSTMModel(train_data, test_data, lstm_hidden_size, time_step, input_dim, output_dim);

%% Run simulation with LSTM-MPC controller
fprintf('Running LSTM-MPC simulation...\n');
[time_lstm, inputs_lstm, outputs_lstm, references_lstm, current_mode_lstm] = runSimulation(lstm_model, modes, mode_switch_time, sim_time, ...
    input_range, delta_u_max, noise_std, setpoint, alpha, beta, T_p, T_c, J_set, max_iter, ...
    learning_rate_0, decay_rate, time_step);

%% Run simulation with traditional MPC controller for comparison
fprintf('Running traditional MPC simulation for comparison...\n');
[time_trad, inputs_trad, outputs_trad, references_trad, current_mode_trad] = traditionalMPC(modes, ...
    mode_switch_time, sim_time, input_range, delta_u_max, noise_std, setpoint, ...
    alpha, beta, T_p, T_c, J_set, max_iter);

%% Run simulation with DNN-MPC controller for comparison
fprintf('Running DNN-MPC simulation for comparison...\n');
[time_dnn, inputs_dnn, outputs_dnn, references_dnn, current_mode_dnn] = dnnMPC(modes, ...
    mode_switch_time, sim_time, input_range, delta_u_max, noise_std, setpoint, ...
    alpha, beta, T_p, T_c, J_set, max_iter);

%% Run simulation with Lyapunov-MPC controller for comparison
fprintf('Running Lyapunov-MPC simulation for comparison...\n');
[time_lyap, inputs_lyap, outputs_lyap, references_lyap, current_mode_lyap] = lyapunovMPC(modes, ...
    mode_switch_time, sim_time, input_range, delta_u_max, noise_std, setpoint, ...
    alpha, beta, T_p, T_c, J_set, max_iter);

%% Calculate performance metrics
% Mean Square Error (MSE)
mse_lstm = mean((outputs_lstm - setpoint).^2);
mse_trad = mean((outputs_trad - setpoint).^2);
mse_dnn = mean((outputs_dnn - setpoint).^2);
mse_lyap = mean((outputs_lyap - setpoint).^2);

% Mean Absolute Error (MAE)
mae_lstm = mean(abs(outputs_lstm - setpoint));
mae_trad = mean(abs(outputs_trad - setpoint));
mae_dnn = mean(abs(outputs_dnn - setpoint));
mae_lyap = mean(abs(outputs_lyap - setpoint));

% Control effort (total variation in control signal)
control_effort_lstm = sum(abs(diff(inputs_lstm)));
control_effort_trad = sum(abs(diff(inputs_trad)));
control_effort_dnn = sum(abs(diff(inputs_dnn)));
control_effort_lyap = sum(abs(diff(inputs_lyap)));

fprintf('\nPerformance Comparison:\n');
fprintf('                     LSTM-MPC    Traditional MPC    DNN-MPC    Lyapunov-MPC\n');
fprintf('MSE:                %8.6f    %8.6f    %8.6f    %8.6f\n', mse_lstm, mse_trad, mse_dnn, mse_lyap);
fprintf('MAE:                %8.6f    %8.6f    %8.6f    %8.6f\n', mae_lstm, mae_trad, mae_dnn, mae_lyap);
fprintf('Control Effort:     %8.6f    %8.6f    %8.6f    %8.6f\n', control_effort_lstm, control_effort_trad, control_effort_dnn, control_effort_lyap);

%% Plot results
fprintf('Plotting results...\n');

% Create a new figure for comparing outputs of all four methods
figure('Name', 'Comparison of All MPC Methods', 'Position', [50, 50, 1200, 800]);

% Plot system outputs
subplot(3, 1, 1);
plot(time_lstm, outputs_lstm, 'b-', 'LineWidth', 1.5);
hold on;
plot(time_trad, outputs_trad, 'r--', 'LineWidth', 1.5);
plot(time_dnn, outputs_dnn, 'g-.', 'LineWidth', 1.5);
plot(time_lyap, outputs_lyap, 'm:', 'LineWidth', 1.5);
plot(time_lstm, references_lstm, 'k-', 'LineWidth', 1);
hold off;
xlabel('Time step');
ylabel('System output');
title('System Response Comparison');
legend('LSTM-MPC', 'Traditional MPC', 'DNN-MPC', 'Lyapunov-MPC', 'Reference');
grid on;

% Plot control inputs
subplot(3, 1, 2);
plot(time_lstm, inputs_lstm, 'b-', 'LineWidth', 1.5);
hold on;
plot(time_trad, inputs_trad, 'r--', 'LineWidth', 1.5);
plot(time_dnn, inputs_dnn, 'g-.', 'LineWidth', 1.5);
plot(time_lyap, inputs_lyap, 'm:', 'LineWidth', 1.5);
hold off;
xlabel('Time step');
ylabel('Control input');
title('Control Signals');
legend('LSTM-MPC', 'Traditional MPC', 'DNN-MPC', 'Lyapunov-MPC');
grid on;

% Plot mode changes
subplot(3, 1, 3);
plot(time_lstm, current_mode_lstm, 'k-', 'LineWidth', 1.5);
xlabel('Time step');
ylabel('System mode');
title('Operating Mode Changes');
yticks(1:length(modes));
ylim([0.5, length(modes)+0.5]);
grid on;

% Highlight mode transition times with vertical lines
for i = 1:length(mode_switch_time)
    line([mode_switch_time(i), mode_switch_time(i)], [0.5, length(modes)+0.5], 'Color', 'r', 'LineStyle', '--');
end

% Create a comparison of tracking errors
figure('Name', 'Tracking Error Comparison', 'Position', [50, 300, 1200, 600]);
plot(time_lstm, abs(outputs_lstm - references_lstm), 'b-', 'LineWidth', 1.5);
hold on;
plot(time_trad, abs(outputs_trad - references_trad), 'r--', 'LineWidth', 1.5);
plot(time_dnn, abs(outputs_dnn - references_dnn), 'g-.', 'LineWidth', 1.5);
plot(time_lyap, abs(outputs_lyap - references_lyap), 'm:', 'LineWidth', 1.5);
hold off;
xlabel('Time step');
ylabel('Absolute tracking error');
title('Tracking Error Comparison');
legend('LSTM-MPC', 'Traditional MPC', 'DNN-MPC', 'Lyapunov-MPC');
grid on;

% Add vertical lines for mode switches
for i = 1:length(mode_switch_time)
    line([mode_switch_time(i), mode_switch_time(i)], get(gca, 'YLim'), 'Color', 'k', 'LineStyle', '--');
end

% Individual plots for each controller
plotResults(time_lstm, inputs_lstm, outputs_lstm, references_lstm, current_mode_lstm, mode_switch_time, 'LSTM-MPC');
plotResults(time_trad, inputs_trad, outputs_trad, references_trad, current_mode_trad, mode_switch_time, 'Traditional MPC');
plotResults(time_dnn, inputs_dnn, outputs_dnn, references_dnn, current_mode_dnn, mode_switch_time, 'DNN-MPC');
plotResults(time_lyap, inputs_lyap, outputs_lyap, references_lyap, current_mode_lyap, mode_switch_time, 'Lyapunov-MPC');

fprintf('Simulation completed.\n'); 