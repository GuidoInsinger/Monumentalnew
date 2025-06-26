### EKF Equations
With state vector

$$\mathbf{x}_k = 
\begin{bmatrix}
x_k \\
y_k \\
\theta_k \\
v_k
\end{bmatrix}$$

* $x$: x position
* $y$: y position
* $\theta$: heading
* $v$: forward velocity in body frame

$$\mathbf{u}_k= 
\begin{bmatrix}
a_{k}^{acc} \\
\omega^{gyro}_k
\end{bmatrix}$$

* $a_{k}^{acc}$: forward acceleration reading at time k
* $\omega^{gyro}_k$: gyroscope reading at time k
* $v$: forward velocity in body frame

This means I ignore the second accelerometer message. This could potentially be useful but I believe it is somewhat redundant information since $a_y=V\omega$ which should, if the rest of the filter does its job, be embedded in the gyroscope measurement and velocities which are updated by the GPS messages

Discretized with Euler integration over timestep $\Delta t$

$$x_{k+1}\'=f(\mathbf{x}_k, \mathbf{u}_k) =
\begin{cases}
x_{k+1} = x_k + v_k \cos(\theta_k) \Delta t \\
y_{k+1} = y_k + v_k \sin(\theta_k) \Delta t \\
\theta_{k+1} &= \theta_k + \omega_k \Delta t \\
v_{k+1} = v_k + a_k^{acc} \Delta t
\end{cases}$$


GPS provides:

$$\mathbf{z}_{k+1} =
\begin{bmatrix}
x_{k+1}^{GPS} \\
y_{k+1}^{GPS}
\end{bmatrix}$$



And the measurement function is:

$$h(\mathbf{x}_{k+1}') =
\begin{bmatrix}
x_{k+1}' \\
y_{k+1}'
\end{bmatrix}$$


### Jacobians
$$F_k =\left\frac{\partial f}{\partial \boldsymbol{x}}\right|_{\boldsymbol{x}=\boldsymbol{x}_k}= 
\begin{bmatrix}
1 & 0 & -v_k \sin(\theta_k) \Delta t & \cos(\theta_k) \Delta t \\\
0 & 1 &  v_k \cos(\theta_k) \Delta t & \sin(\theta_k) \Delta t \\\
0 & 0 & 1 & 0 \\\
0 & 0 & 0 & 1
\end{bmatrix}$$

$$H_{k+1} = \left\frac{\partial f}{\partial \boldsymbol{x}}\right|_{\boldsymbol{x}=\boldsymbol{x}_{k+1}'} =
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0
\end{bmatrix}$$



#### Prediction Step

$$\mathbf{x}_{k+1}' = f(\hat{\mathbf{x}}_{k})$$

$$P_{k+1}' = F_k P_k F_k^\top + Q$$

Update Step (if GPS available):

$$\mathbf{\hat{y}}_{k+1} = \mathbf{z}_{k+1} - h(\mathbf{x}_{k+1})$$

$$S_{k+1} = K_{k+1} P_{k+1}' H_{k+1}^\top + R$$

$$K_{k+1} = P_{k+1}' H_{k+1}^\top S_{k+1}^{-1}$$

$$\mathbf{x}_{k+1} = \mathbf{x}_{k+1}' + K_{k+1} \mathbf{y}_{k+1}$$

$$P_{k+1} = (I - K_{k+1} H_{k+1}) P_{k+1}'$$

