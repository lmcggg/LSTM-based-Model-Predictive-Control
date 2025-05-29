# LSTM-MPC: Deep Learning Based Predictive Control for Multimode Process Control

This project implements a Long Short-Term Memory based Model Predictive Control (LSTM-MPC) method specifically designed for process control systems with multimodal dynamic characteristics. By combining deep learning with traditional Model Predictive Control (MPC), LSTM-MPC effectively handles system mode changes, improving control performance and robustness.

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

## 6. System Requirements

- MATLAB R2019b or later
- Deep Learning Toolbox

## 7. References

1. Li S, Wang D, Zhou Y, Liu Y. LSTM-MPC: A Deep Learning Based Predictive Control Method for Multimode Process Control. IEEE Transactions on Neural Networks and Learning Systems.
2. Rawlings JB, Mayne DQ, Diehl M. Model Predictive Control: Theory, Computation, and Design. 2nd ed. Nob Hill Publishing, 2017.
3. Hochreiter S, Schmidhuber J. Long Short-Term Memory. Neural Computation, 1997, 9(8): 1735-1780.

---

# LSTM-MPC: 基于深度学习的多模态过程控制预测方法

该项目实现了一种基于长短期记忆网络的模型预测控制方法（LSTM-MPC），专门用于处理具有多模态动态特性的过程控制系统。LSTM-MPC 通过将深度学习与传统模型预测控制（MPC）相结合，可以有效应对系统模态变化，提高控制性能和鲁棒性。

## 1. 系统概述

### 1.1 Hammerstein 系统模型

本项目使用 Hammerstein 系统作为研究对象，该系统由非线性静态部分和线性动态部分组成。系统具有多个运行模态，在不同的模态下具有不同的参数值。

系统数学模型如下：

**非线性静态部分**:
$$x(k) = a_m \cdot u(k) - b_m \cdot u(k)^2 + 0.5 \cdot u(k)^3$$

**线性动态部分**:
$$y(k+1) = 0.6 \cdot y(k) - 0.1 \cdot y(k-1) + 1.2 \cdot x(k) - 0.1 \cdot x(k-1) + v(k)$$

其中：
- $u(k)$ 是系统输入
- $y(k)$ 是系统输出
- $x(k)$ 是中间状态变量
- $a_m$ 和 $b_m$ 是与模态相关的参数
- $v(k)$ 是均值为 0，标准差为 $\sigma$ 的高斯白噪声

本项目中模拟了三种不同的模态：
- 模态 1: $a_m = 1.5$, $b_m = 1.5$
- 模态 2: $a_m = 1.0$, $b_m = 1.0$
- 模态 3: $a_m = 0.5$, $b_m = 0.5$

### 1.2 控制目标

控制目标是使系统输出 $y(k)$ 跟踪给定的设定值（setpoint），同时考虑输入约束和输入变化率约束。

## 2. LSTM-MPC 方法原理

### 2.1 LSTM 预测模型

LSTM（Long Short-Term Memory）是一种特殊的循环神经网络（RNN），能够学习长期依赖关系。在本项目中，LSTM 被用作系统的预测模型，它能够根据当前和历史的输入输出数据预测未来的系统输出。

LSTM 模型的输入为 $[u(k), u(k-1), y(k), y(k-1)]$，输出为 $\hat{y}(k+1)$，即系统的一步预测。

模型训练过程可以表示为最小化预测误差：

$$\min_\theta \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

其中 $\theta$ 是 LSTM 网络的参数，$N$ 是训练样本数量。

### 2.2 基于 LSTM 的模型预测控制

LSTM-MPC 方法的核心是将 LSTM 预测模型整合到 MPC 框架中。MPC 优化问题可以表述为：

$$\min_{{\Delta u(k|k), \Delta u(k+1|k), ..., \Delta u(k+T_c-1|k)}} J(k)$$

其中目标函数 $J(k)$ 定义为：

$$J(k) = \sum_{j=1}^{T_p} \alpha [r(k+j) - \hat{y}(k+j|k)]^2 + \sum_{j=0}^{T_c-1} \beta [\Delta u(k+j|k)]^2$$

约束条件：
$$u_{min} \leq u(k+j|k) \leq u_{max}, \quad j = 0, 1, ..., T_c-1$$
$$-\Delta u_{max} \leq \Delta u(k+j|k) \leq \Delta u_{max}, \quad j = 0, 1, ..., T_c-1$$

其中：
- $\Delta u(k+j|k)$ 是在时刻 $k$ 预测的 $k+j$ 时刻的控制增量
- $\hat{y}(k+j|k)$ 是在时刻 $k$ 预测的 $k+j$ 时刻的系统输出
- $r(k+j)$ 是 $k+j$ 时刻的参考设定值
- $T_p$ 是预测时域
- $T_c$ 是控制时域
- $\alpha$ 是跟踪误差权重
- $\beta$ 是控制增量权重

### 2.3 基于 LSTM 的多步预测

多步预测是通过迭代使用 LSTM 模型实现的。在时刻 $k$，有：

1. 使用当前输入序列 $[u(k), u(k-1), y(k), y(k-1)]$ 预测 $\hat{y}(k+1|k)$
2. 更新输入序列为 $[u(k+1|k), u(k), \hat{y}(k+1|k), y(k)]$
3. 预测 $\hat{y}(k+2|k)$
4. 依此类推，直到预测时域 $T_p$

### 2.4 雅可比矩阵计算

为了使用基于梯度的优化方法，需要计算预测输出对控制增量的雅可比矩阵：

$$J(k) = \begin{bmatrix} 
\frac{\partial \hat{y}(k+1|k)}{\partial \Delta u(k|k)} & \frac{\partial \hat{y}(k+1|k)}{\partial \Delta u(k+1|k)} & \cdots & \frac{\partial \hat{y}(k+1|k)}{\partial \Delta u(k+T_c-1|k)} \\
\frac{\partial \hat{y}(k+2|k)}{\partial \Delta u(k|k)} & \frac{\partial \hat{y}(k+2|k)}{\partial \Delta u(k+1|k)} & \cdots & \frac{\partial \hat{y}(k+2|k)}{\partial \Delta u(k+T_c-1|k)} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial \hat{y}(k+T_p|k)}{\partial \Delta u(k|k)} & \frac{\partial \hat{y}(k+T_p|k)}{\partial \Delta u(k+1|k)} & \cdots & \frac{\partial \hat{y}(k+T_p|k)}{\partial \Delta u(k+T_c-1|k)} \\
\end{bmatrix}$$

在本项目中，采用数值微分方法计算雅可比矩阵：

$$\frac{\partial \hat{y}(k+i|k)}{\partial \Delta u(k+j|k)} \approx \frac{\hat{y}(k+i|k, \Delta u(k+j|k) + \delta) - \hat{y}(k+i|k, \Delta u(k+j|k))}{\delta}$$

其中 $\delta$ 是一个小的扰动值。

### 2.5 自适应投影梯度下降优化

MPC 优化问题使用自适应投影梯度下降方法求解。

梯度计算：
$$\nabla J(k) = -\alpha J(k)^T [r(k) - \hat{y}(k)] + \beta \Delta U(k)$$

控制增量更新：
$$\Delta U(k+1) = \Delta U(k) - \frac{\eta(k)}{1 + \eta(k) \beta} [\nabla J(k)]$$

学习率自适应更新：
$$\eta(k+1) = \eta(k) e^{-\lambda k}$$

投影操作：
$$\Delta u(k+j|k) = \min(\Delta u_{max}, \max(-\Delta u_{max}, \Delta u(k+j|k)))$$
$$u(k+j|k) = \min(u_{max}, \max(u_{min}, u(k+j-1|k) + \Delta u(k+j|k)))$$

其中：
- $\eta(k)$ 是学习率
- $\lambda$ 是学习率衰减系数

## 3. 代码实现

### 3.1 文件结构

- `main.m`: 主脚本，运行整个仿真
- `hammersteinSystem.m`: Hammerstein 系统模型实现
- `collectTrainingData.m`: 收集 LSTM 模型训练数据
- `trainLSTMModel.m`: 训练 LSTM 模型
- `predictLSTM.m`: 使用训练好的 LSTM 模型进行预测
- `predictMultiStep.m`: 使用 LSTM 模型进行多步预测
- `calculateJacobian.m`: 计算雅可比矩阵
- `mpcOptimization.m`: 实现 MPC 优化算法
- `runSimulation.m`: 运行 LSTM-MPC 控制系统仿真
- `traditionalMPC.m`: 传统 MPC 方法实现（用于比较）
- `dnnMPC.m`: 基于 DNN 的 MPC 方法实现（用于比较）
- `lyapunovMPC.m`: 基于 Lyapunov 的 MPC 方法实现（用于比较）
- `plotResults.m`: 绘制仿真结果

### 3.2 主要参数

- LSTM 模型参数：
  - 隐藏层大小: 16
  - 时间步长: 2
  - 输入维度: 4 ([u(t), u(t-1), y(t), y(t-1)])
  - 输出维度: 1 (y(t+1))

- MPC 参数：
  - 跟踪误差权重 $\alpha$: 1
  - 控制增量权重 $\beta$: 1
  - 预测时域 $T_p$: 2
  - 控制时域 $T_c$: 1
  - 最大控制增量 $\Delta u_{max}$: 0.1
  - 优化终止阈值: 1e-4
  - 最大迭代次数: 50
  - 初始学习率: 0.01
  - 学习率衰减系数: 0.05

## 4. 使用方法

1. 运行 `main.m` 脚本开始仿真：
   - 生成训练数据
   - 训练 LSTM 模型
   - 运行 LSTM-MPC 控制系统仿真
   - 运行传统 MPC、DNN-MPC 和 Lyapunov-MPC 进行比较
   - 计算性能指标
   - 绘制结果

2. 查看仿真结果：
   - 系统输出跟踪性能
   - 控制输入
   - 模态切换情况
   - 跟踪误差对比

## 5. 性能评估

LSTM-MPC 方法与传统 MPC、DNN-MPC 和 Lyapunov-MPC 方法进行了比较，使用以下性能指标：

1. 均方误差 (MSE)：
   $$MSE = \frac{1}{N}\sum_{k=1}^{N}(y(k) - r(k))^2$$

2. 平均绝对误差 (MAE)：
   $$MAE = \frac{1}{N}\sum_{k=1}^{N}|y(k) - r(k)|$$

3. 控制效果（控制信号的总变化）：
   $$\sum_{k=2}^{N}|u(k) - u(k-1)|$$

## 6. 系统要求

- MATLAB R2019b 或更高版本
- Deep Learning Toolbox

## 7. 参考文献

1. Li S, Wang D, Zhou Y, Liu Y. LSTM-MPC: A Deep Learning Based Predictive Control Method for Multimode Process Control. IEEE Transactions on Neural Networks and Learning Systems.
2. Rawlings JB, Mayne DQ, Diehl M. Model Predictive Control: Theory, Computation, and Design. 2nd ed. Nob Hill Publishing, 2017.
3. Hochreiter S, Schmidhuber J. Long Short-Term Memory. Neural Computation, 1997, 9(8): 1735-1780. 