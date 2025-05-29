function plotResults(time, inputs, outputs, references, current_mode, mode_switch_time, controller_name)
% PLOTRESULTS Plots the results of the LSTM-MPC simulation
%
% Inputs:
%   time            - Time vector
%   inputs          - Control inputs
%   outputs         - System outputs
%   references      - Reference setpoints
%   current_mode    - Current system mode at each time step
%   mode_switch_time- Time steps where mode switches occur
%   controller_name - Name of the controller used (optional)

% Set default controller name if not provided
if nargin < 7
    controller_name = 'MPC';
end

figure('Position', [100, 100, 1000, 600], 'Name', [controller_name, ' Results']);

% Plot system output and reference
subplot(3, 1, 1);
plot(time, outputs, 'b-', 'LineWidth', 1.5);
hold on;
plot(time, references, 'r--', 'LineWidth', 1.5);
% Add vertical lines for mode switches
for i = 1:length(mode_switch_time)
    xline(mode_switch_time(i), 'k--');
    text(mode_switch_time(i) + 2, max(outputs), ['Mode ' num2str(i)], 'FontSize', 10);
end
grid on;
ylabel('Output y(t)');
title([controller_name, ' Control Performance']);
legend('System Output', 'Reference', 'Location', 'Best');

% Plot control input
subplot(3, 1, 2);
plot(time, inputs, 'g-', 'LineWidth', 1.5);
hold on;
% Add vertical lines for mode switches
for i = 1:length(mode_switch_time)
    xline(mode_switch_time(i), 'k--');
end
grid on;
ylabel('Control Input u(t)');

% Plot tracking error
subplot(3, 1, 3);
tracking_error = outputs - references;
plot(time, tracking_error, 'm-', 'LineWidth', 1.5);
hold on;
% Add vertical lines for mode switches
for i = 1:length(mode_switch_time)
    xline(mode_switch_time(i), 'k--');
end
grid on;
xlabel('Time Step');
ylabel('Tracking Error');

% Calculate performance metrics
mse = mean(tracking_error.^2);
mae = mean(abs(tracking_error));

% Display performance metrics
text(0.7*max(time), 0.8*max(abs(tracking_error)), ['MSE: ' num2str(mse, '%.4e')], 'FontSize', 10);
text(0.7*max(time), 0.6*max(abs(tracking_error)), ['MAE: ' num2str(mae, '%.4e')], 'FontSize', 10);

end 