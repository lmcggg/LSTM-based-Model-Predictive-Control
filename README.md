# LSTM-MPC: Deep Learning Based Predictive Control for Multimode Process Control

This project implements a Long Short-Term Memory based Model Predictive Control (LSTM-MPC) method specifically designed for process control systems with multimodal dynamic characteristics. By combining deep learning with traditional Model Predictive Control (MPC), LSTM-MPC effectively handles system mode changes, improving control performance and robustness.this method is from the paper 'LSTM-MPC: A Deep Learning Based Predictive Control Method for Multimode Process Control'-IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS, VOL. 70, NO. 11, NOVEMBER 2023 (Keke Huang , Member, IEEE, Ke Wei, Fanbiao Li, Chunhua Yang , Fellow, IEEE, and Weihua Gui)

## 1. System Overview

### 1.1 Hammerstein System Model

This project uses a Hammerstein system as the research object, which consists of a nonlinear static part and a linear dynamic part. The system has multiple operating modes with different parameter values in each mode.

The mathematical model of the system is as follows:

**Nonlinear Static Part**:
$$x(k) = a_m \cdot u(k) - b_m \cdot u(k)^2 + 0.5 \cdot u(k)^3$$

**Linear Dynamic Part**:
$$y(k+1) = 0.6 \cdot y(k) - 0.1 \cdot y(k-1) + 1.2 \cdot x(k) - 0.1 \cdot x(k-1) + v(k)$$

Where:
- $u(k)$ is the system input
- $y(k)$ is the system output
- $x(k)$ is the intermediate state variable
- $a_m$ and $b_m$ are mode-related parameters
- $v(k)$ is Gaussian white noise with mean 0 and standard deviation $\sigma$

Three different modes are simulated in this project:
- Mode 1: $a_m = 1.5$, $b_m = 1.5$
- Mode 2: $a_m = 1.0$, $b_m = 1.0$
- Mode 3: $a_m = 0.5$, $b_m = 0.5$

### 1.2 Control Objective

The control objective is to make the system output $y(k)$ track a given setpoint while considering input constraints and input rate constraints.

## 2. LSTM-MPC Method Principles

### 2.1 LSTM Prediction Model

Long Short-Term Memory (LSTM) is a special type of recurrent neural network (RNN) capable of learning long-term dependencies. In this project, LSTM is used as the system's prediction model, which can predict future system outputs based on current and historical input-output data.

The LSTM model takes $[u(k), u(k-1), y(k), y(k-1)]$ as input and outputs $\hat{y}(k+1)$, which is the one-step prediction of the system.

The model training process can be represented as minimizing the prediction error:

$$\min_\theta \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

Where $\theta$ represents the parameters of the LSTM network and $N$ is the number of training samples.

### 2.2 LSTM-based Model Predictive Control

The core of the LSTM-MPC method is to integrate the LSTM prediction model into the MPC framework. The MPC optimization problem can be formulated as:

$$\min_{{\Delta u(k|k), \Delta u(k+1|k), ..., \Delta u(k+T_c-1|k)}} J(k)$$

Where the objective function $J(k)$ is defined as:

$$J(k) = \sum_{j=1}^{T_p} \alpha [r(k+j) - \hat{y}(k+j|k)]^2 + \sum_{j=0}^{T_c-1} \beta [\Delta u(k+j|k)]^2$$

Subject to constraints:
$$u_{min} \leq u(k+j|k) \leq u_{max}, \quad j = 0, 1, ..., T_c-1$$
$$-\Delta u_{max} \leq \Delta u(k+j|k) \leq \Delta u_{max}, \quad j = 0, 1, ..., T_c-1$$

Where:
- $\Delta u(k+j|k)$ is the control increment at time $k+j$ predicted at time $k$
- $\hat{y}(k+j|k)$ is the system output at time $k+j$ predicted at time $k$
- $r(k+j)$ is the reference setpoint at time $k+j$
- $T_p$ is the prediction horizon
- $T_c$ is the control horizon
- $\alpha$ is the weight for tracking error
- $\beta$ is the weight for control increment

### 2.3 LSTM-based Multi-step Prediction

Multi-step prediction is implemented by iteratively using the LSTM model. At time $k$:

1. Use the current input sequence $[u(k), u(k-1), y(k), y(k-1)]$ to predict $\hat{y}(k+1|k)$
2. Update the input sequence to $[u(k+1|k), u(k), \hat{y}(k+1|k), y(k)]$
3. Predict $\hat{y}(k+2|k)$
4. Continue this process until the prediction horizon $T_p$

### 2.4 Jacobian Matrix Calculation

To use gradient-based optimization methods, it is necessary to calculate the Jacobian matrix of predicted outputs with respect to control increments:

$$J(k) = \begin{bmatrix} 
\frac{\partial \hat{y}(k+1|k)}{\partial \Delta u(k|k)} & \frac{\partial \hat{y}(k+1|k)}{\partial \Delta u(k+1|k)} & \cdots & \frac{\partial \hat{y}(k+1|k)}{\partial \Delta u(k+T_c-1|k)} \\
\frac{\partial \hat{y}(k+2|k)}{\partial \Delta u(k|k)} & \frac{\partial \hat{y}(k+2|k)}{\partial \Delta u(k+1|k)} & \cdots & \frac{\partial \hat{y}(k+2|k)}{\partial \Delta u(k+T_c-1|k)} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial \hat{y}(k+T_p|k)}{\partial \Delta u(k|k)} & \frac{\partial \hat{y}(k+T_p|k)}{\partial \Delta u(k+1|k)} & \cdots & \frac{\partial \hat{y}(k+T_p|k)}{\partial \Delta u(k+T_c-1|k)} \\
\end{bmatrix}$$

In this project, numerical differentiation is used to calculate the Jacobian matrix:

$$\frac{\partial \hat{y}(k+i|k)}{\partial \Delta u(k+j|k)} \approx \frac{\hat{y}(k+i|k, \Delta u(k+j|k) + \delta) - \hat{y}(k+i|k, \Delta u(k+j|k))}{\delta}$$

Where $\delta$ is a small perturbation value.

### 2.5 Adaptive Projected Gradient Descent Optimization

The MPC optimization problem is solved using adaptive projected gradient descent.

Gradient calculation:
$$\nabla J(k) = -\alpha J(k)^T [r(k) - \hat{y}(k)] + \beta \Delta U(k)$$

Control increment update:
$$\Delta U(k+1) = \Delta U(k) - \frac{\eta(k)}{1 + \eta(k) \beta} [\nabla J(k)]$$

Adaptive learning rate update:
$$\eta(k+1) = \eta(k) e^{-\lambda k}$$

Projection operation:
$$\Delta u(k+j|k) = \min(\Delta u_{max}, \max(-\Delta u_{max}, \Delta u(k+j|k)))$$
$$u(k+j|k) = \min(u_{max}, \max(u_{min}, u(k+j-1|k) + \Delta u(k+j|k)))$$

Where:
- $\eta(k)$ is the learning rate
- $\lambda$ is the learning rate decay coefficient

## 3. Code Implementation

### 3.1 File Structure

- `main.m`: Main script that runs the entire simulation
- `hammersteinSystem.m`: Implementation of the Hammerstein system model
- `collectTrainingData.m`: Collects training data for the LSTM model
- `trainLSTMModel.m`: Trains the LSTM model
- `predictLSTM.m`: Makes predictions using the trained LSTM model
- `predictMultiStep.m`: Performs multi-step prediction using the LSTM model
- `calculateJacobian.m`: Calculates the Jacobian matrix
- `mpcOptimization.m`: Implements the MPC optimization algorithm
- `runSimulation.m`: Runs the LSTM-MPC control system simulation
- `traditionalMPC.m`: Implementation of traditional MPC method (for comparison)
- `dnnMPC.m`: Implementation of DNN-based MPC method (for comparison)
- `lyapunovMPC.m`: Implementation of Lyapunov-based MPC method (for comparison)
- `plotResults.m`: Plots the simulation results

### 3.2 Key Parameters

- LSTM model parameters:
  - Hidden layer size: 16
  - Time step: 2
  - Input dimension: 4 ([u(t), u(t-1), y(t), y(t-1)])
  - Output dimension: 1 (y(t+1))

- MPC parameters:
  - Tracking error weight $\alpha$: 1
  - Control increment weight $\beta$: 1
  - Prediction horizon $T_p$: 2
  - Control horizon $T_c$: 1
  - Maximum control increment $\Delta u_{max}$: 0.1
  - Optimization termination threshold: 1e-4
  - Maximum iteration count: 50
  - Initial learning rate: 0.01
  - Learning rate decay coefficient: 0.05

## 4. Usage

1. Run the `main.m` script to start the simulation:
   - Generate training data
   - Train the LSTM model
   - Run the LSTM-MPC control system simulation
   - Run traditional MPC, DNN-MPC, and Lyapunov-MPC for comparison
   - Calculate performance metrics
   - Plot results

2. View the simulation results:
   - System output tracking performance
   - Control inputs
   - Mode switching
   - Tracking error comparison

## 5. Performance Evaluation

The LSTM-MPC method is compared with traditional MPC, DNN-MPC, and Lyapunov-MPC methods using the following performance metrics:

1. Mean Square Error (MSE):
   $$MSE = \frac{1}{N}\sum_{k=1}^{N}(y(k) - r(k))^2$$

2. Mean Absolute Error (MAE):
   $$MAE = \frac{1}{N}\sum_{k=1}^{N}|y(k) - r(k)|$$

3. Control effort (total variation in control signal):
   $$\sum_{k=2}^{N}|u(k) - u(k-1)|$$

## 6. Result

![image](https://github.com/user-attachments/assets/205ca875-ee55-4c17-bad9-681be662c08a)

![image](https://github.com/user-attachments/assets/a52e1924-212b-4591-8d88-da1286f0b0b5)

![image](https://github.com/user-attachments/assets/60313c01-5ccf-4e17-b077-00e40371c049)

![image](https://github.com/user-attachments/assets/9ac00fca-ccc1-42f5-bb23-648feafb785b)


Performance Comparison:

                     LSTM-MPC    Traditional MPC    DNN-MPC    Lyapunov-MPC

MSE:                0.004558    0.249597    0.004060    0.005443

MAE:                0.024741    0.498646    0.027413    0.042637

Control Effort:     1.261174    0.500000    2.797715    10.089449


## 6. System Requirements

- MATLAB R2019b or later
- Deep Learning Toolbox

