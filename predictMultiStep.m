function y_pred_sequence = predictMultiStep(lstm_model, input_sequence, u_increments)
% PREDICTMULTISTEP Predict multiple steps ahead using the LSTM model
% 
% Inputs:
%   lstm_model      - Trained LSTM model
%   input_sequence  - Current input sequence [u(t), u(t-1), y(t), y(t-1)]
%   u_increments    - Vector of control increments Δu(t), Δu(t+1), ...
%
% Output:
%   y_pred_sequence - Sequence of predicted outputs [y(t+1), y(t+2), ...]

control_horizon = length(u_increments);
prediction_horizon = 2; % As per the paper
y_pred_sequence = zeros(prediction_horizon, 1);

% Extract current values from input sequence
u_current = input_sequence(1);
u_prev = input_sequence(2);
y_current = input_sequence(3);
y_prev = input_sequence(4);

% Compute first prediction
u_new = u_current + u_increments(1);
x_new = [u_new, u_current, y_current, y_prev];
y_next = predictLSTM(lstm_model, x_new);
y_pred_sequence(1) = y_next;

% If prediction horizon is greater than 1, continue predictions
if prediction_horizon > 1
    for i = 2:prediction_horizon
        u_idx = min(i, control_horizon);
        u_new_next = u_new + u_increments(u_idx);
        x_new_next = [u_new_next, u_new, y_next, y_current];
        y_next_next = predictLSTM(lstm_model, x_new_next);
        y_pred_sequence(i) = y_next_next;
        
        % Update for next iteration
        u_prev = u_new;
        u_new = u_new_next;
        y_prev = y_current;
        y_current = y_next;
        y_next = y_next_next;
    end
end

end 