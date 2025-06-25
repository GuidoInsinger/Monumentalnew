$$
\mathbf{x} = 
\begin{bmatrix}
x \\
y \\
\theta \\
v \\
\end{bmatrix}
$$

* $x$: x position
* $y$: y position
* $\theta$: heading
* $v$: forward velocity in body frame



Discretized with Euler integration over timestep $\Delta t$:

$$
\frac{\partial f}{\partial \mathbf{x}} =
\begin{aligned}
x_{k+1} &= x_k + v_k \cos(\theta_k) \Delta t \\
y_{k+1} &= y_k + v_k \sin(\theta_k) \Delta t \\
\theta_{k+1} &= \theta_k + \omega_k \Delta t \\
v_{k+1} &= v_k + a_k \Delta t \\
\end{aligned}
$$

Note:

* The yaw rate is updated toward the measured value.
* The acceleration is assumed to directly change the forward speed.


Linearized around the current estimate:

$$
F_k = \frac{\partial f}{\partial \mathbf{x}} = 
\begin{bmatrix}
1 & 0 & -v \sin(\theta) \Delta t & \cos(\theta) \Delta t & \\
0 & 1 &  v \cos(\theta) \Delta t & \sin(\theta) \Delta t & \\
0 & 0 & 1 & 0 & \\
0 & 0 & 0 & 1 & \\

\end{bmatrix}
$$


GPS provides:

$$
\mathbf{z}_k =
\begin{bmatrix}
x_k \\
y_k
\end{bmatrix}
$$

So the measurement function is:

$$
h(\mathbf{x}) =
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

And its Jacobian is:

$$
H_k = \frac{\partial h}{\partial \mathbf{x}} =
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0
\end{bmatrix}
$$

---

## 🔷 6. **EKF Equations**

### 🧭 Prediction Step:

$$
\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1})
$$

$$
P_{k|k-1} = F_k P_{k-1} F_k^\top + Q
$$

### 📡 Update Step (if GPS available):

$$
\mathbf{y}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})
$$

$$
S_k = H_k P_{k|k-1} H_k^\top + R
$$

$$
K_k = P_{k|k-1} H_k^\top S_k^{-1}
$$

$$
\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_{k|k-1} + K_k \mathbf{y}_k
$$

$$
P_k = (I - K_k H_k) P_{k|k-1}
$$

