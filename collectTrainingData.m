function [train_data, test_data] = collectTrainingData(modes, input_range, noise_std)
% COLLECTTRAININGDATA Collects training and testing data for LSTM model
%
% Inputs:
%   modes       - Array of system modes containing a_m and b_m parameters
%   input_range - Range of control inputs [min, max]
%   noise_std   - Standard deviation of Gaussian noise
%
% Outputs:
%   train_data  - Structure containing training data
%   test_data   - Structure containing testing data

% Parameters
num_modes = length(modes);
data_points_per_mode = 1000;
total_points = num_modes * data_points_per_mode;

% Initialize data arrays
X_data = zeros(total_points, 4);  % [u(t), u(t-1), y(t), y(t-1)]
Y_data = zeros(total_points, 1);  % y(t+1)

% Generate random inputs for training within the given range
u_min = input_range(1);
u_max = input_range(2);

data_idx = 1;
for mode_idx = 1:num_modes
    current_mode = modes(mode_idx);
    
    % Initialize for this mode - add extra elements for proper indexing
    u_seq = u_min + (u_max - u_min) * rand(data_points_per_mode + 3, 1);
    y_seq = zeros(data_points_per_mode + 3, 1);
    
    % Initial conditions for the first 3 elements
    % These will serve as the initial values for the system
    y_seq(1:3) = 0;
    
    % Generate sequence for this mode
    for i = 4:length(y_seq)
        y_seq(i) = hammersteinSystem(u_seq(i-1), u_seq(i-2), y_seq(i-1), y_seq(i-2), current_mode, noise_std);
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

% Split into training (80%) and testing (20%) sets
train_ratio = 0.8;
train_size = round(total_points * train_ratio);

train_data = struct('X', X_data(1:train_size, :), 'Y', Y_data(1:train_size, :));
test_data = struct('X', X_data(train_size+1:end, :), 'Y', Y_data(train_size+1:end, :));

end 