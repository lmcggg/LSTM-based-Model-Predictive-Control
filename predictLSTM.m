function y_pred = predictLSTM(lstm_model, input_sequence)
% PREDICTLSTM Makes a prediction using the trained LSTM model
%
% Inputs:
%   lstm_model     - Trained LSTM model structure with network and normalization params
%   input_sequence - Input sequence [u(t), u(t-1), y(t), y(t-1)]
%
% Output:
%   y_pred         - Predicted output y(t+1)

% Extract model components
net = lstm_model.net;
mu_X = lstm_model.mu_X;
sigma_X = lstm_model.sigma_X;
mu_Y = lstm_model.mu_Y;
sigma_Y = lstm_model.sigma_Y;

% Normalize input
input_norm = (input_sequence - mu_X) ./ sigma_X;

% Create cell array for network input (similar to training format)
input_cell = {input_norm'};  % Column vector as required by network

% Make prediction
y_pred_norm = predict(net, input_cell);

% Denormalize output
y_pred = y_pred_norm * sigma_Y + mu_Y;

end 