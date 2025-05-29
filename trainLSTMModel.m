function lstm_model = trainLSTMModel(train_data, test_data, hidden_size, time_step, input_dim, output_dim)
% TRAINLSTMMODEL Trains an LSTM model for predictive control
%
% Inputs:
%   train_data  - Structure containing training data
%   test_data   - Structure containing testing data
%   hidden_size - Number of hidden units in the LSTM layer
%   time_step   - Sequence length for LSTM input
%   input_dim   - Input dimension
%   output_dim  - Output dimension
%
% Output:
%   lstm_model  - Trained LSTM network

% Extract training and testing data
X_train = train_data.X;
Y_train = train_data.Y;
X_test = test_data.X;
Y_test = test_data.Y;

% Normalize data
[X_train_norm, mu_X, sigma_X] = normalize_data(X_train);
[Y_train_norm, mu_Y, sigma_Y] = normalize_data(Y_train);
X_test_norm = (X_test - mu_X) ./ sigma_X;
Y_test_norm = (Y_test - mu_Y) ./ sigma_Y;

% Create and configure LSTM network
layers = [
    sequenceInputLayer(input_dim)
    lstmLayer(hidden_size, 'OutputMode', 'last')
    fullyConnectedLayer(output_dim)
    regressionLayer
];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', {X_test_norm, Y_test_norm}, ...
    'ValidationFrequency', 30);

% Format data as cell arrays of sequences
numObservations = size(X_train_norm, 1);
numTestObservations = size(X_test_norm, 1);

X_train_seq = cell(numObservations, 1);
for i = 1:numObservations
    X_train_seq{i} = X_train_norm(i,:)';
end

X_test_seq = cell(numTestObservations, 1);
for i = 1:numTestObservations
    X_test_seq{i} = X_test_norm(i,:)';
end

% Update validation data format
options.ValidationData = {X_test_seq, Y_test_norm};

% Train the network
net = trainNetwork(X_train_seq, Y_train_norm, layers, options);

% Package the trained model with normalization parameters
lstm_model = struct('net', net, 'mu_X', mu_X, 'sigma_X', sigma_X, 'mu_Y', mu_Y, 'sigma_Y', sigma_Y);
end

function [normalized_data, mu, sigma] = normalize_data(data)
% Normalize data to have zero mean and unit variance
mu = mean(data);
sigma = std(data);
% Avoid division by zero
sigma(sigma < eps) = 1;
normalized_data = (data - mu) ./ sigma;
end 