function J = calculateJacobian(lstm_model, input_sequence, control_horizon, delta)
% CALCULATEJACOBIAN Calculates the Jacobian matrix of the LSTM predictions
% with respect to control inputs using numerical differentiation
%
% Inputs:
%   lstm_model      - Trained LSTM model
%   input_sequence  - Current input sequence [u(t), u(t-1), y(t), y(t-1)]
%   control_horizon - Number of future control steps to optimize
%   delta           - Small perturbation for numerical differentiation
%
% Output:
%   J               - Jacobian matrix (prediction_horizon x control_horizon)

% For LSTM-MPC, we need the Jacobian of future outputs w.r.t future inputs
% The paper suggests using numerical differentiation
prediction_horizon = 2; % As per the paper
J = zeros(prediction_horizon, control_horizon);

% Base prediction without perturbation
y_base = predictMultiStep(lstm_model, input_sequence, zeros(control_horizon, 1));

% Calculate Jacobian for each control input and prediction step
for j = 1:control_horizon
    % Create perturbed control sequence
    u_perturb = zeros(control_horizon, 1);
    u_perturb(j) = delta;
    
    % Predict with perturbation
    y_perturb = predictMultiStep(lstm_model, input_sequence, u_perturb);
    
    % Calculate partial derivatives
    for i = 1:prediction_horizon
        J(i, j) = (y_perturb(i) - y_base(i)) / delta;
    end
end

end 