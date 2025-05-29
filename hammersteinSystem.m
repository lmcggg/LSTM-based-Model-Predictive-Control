function y_next = hammersteinSystem(u_current, u_prev, y_current, y_prev, mode, noise_std)
% HAMMERSTEINSYSTEM Simulates the Hammerstein system with multimode behavior
%
% Inputs:
%   u_current - Current control input
%   u_prev    - Previous control input
%   y_current - Current system output
%   y_prev    - Previous system output
%   mode      - Current system mode containing a_m and b_m parameters
%   noise_std - Standard deviation of Gaussian noise
%
% Output:
%   y_next    - Next system output

% Hammerstein system equations (Eq. 42)
% x(k) = a_m*u(k) - b_m*u(k)^2 + 0.5*u(k)^3
% y(k+1) = 0.6*y(k) - 0.1*y(k-1) + 1.2*x(k) - 0.1*x(k-1) + v(k)

% Calculate nonlinear static part for current and previous inputs
x_current = mode.a_m * u_current - mode.b_m * u_current^2 + 0.5 * u_current^3;
x_prev = mode.a_m * u_prev - mode.b_m * u_prev^2 + 0.5 * u_prev^3;

% Add random Gaussian noise
noise = noise_std * randn();

% Calculate next output using the dynamic linear part
y_next = 0.6 * y_current - 0.1 * y_prev + 1.2 * x_current - 0.1 * x_prev + noise;

end 