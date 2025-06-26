# Differential drive control

To estimate the state of the robot and deal with the asynchronous measurements I chose an EKF. In my EKF I use the inertial measurements as inputs to the prediction step, and the GPS measurements as inputs for the update step. This allows me to run the prediction step whenever a new inertial measurement comes in which I found to be around ~20 hz and an update step to correct the predictions using the GPS measurement at around ~1hz. The uncertainty propagation ensures the EKF covariance will correctly encapsulate the compounded uncertainies from the ~20 prediction steps happening up until the point a GPS measurement comes in. 

Another reason for why I chose to use inertial measurements in my update step rather than the true inputs is that I found the actuator dynamics to be quite noisy and with no explicit noise model given. Setting up a proper model to track this seemed more complicated than a simple kinematic estimator, the benefit of which seems questionable. 


## EKF Equations 

* $x$: x position
* $y$: y position
* $\theta$: heading
* $v$: forward velocity in body frame

```math
\mathbf{x}_k = 
\begin{bmatrix}
x_k \\
y_k \\
\theta_k \\
v_k
\end{bmatrix}
```

* $a_{k}^{acc}$: forward acceleration reading at time k
* $\omega^{gyro}_k$: gyroscope reading at time k

```math
\mathbf{u}_k= 
\begin{bmatrix}
a_{k}^{acc} \\
\omega^{gyro}_k
\end{bmatrix}
```


This means I ignore the second accelerometer message. This could potentially be useful but I believe it is somewhat redundant information since $a_y=V\omega$ which should, if the rest of the filter does its job, be embedded in the gyroscope measurement and velocities which are updated by the GPS messages

---
Discretized with Euler integration over timestep $\Delta t$, where I get the value of $\Delta t$ from the arrival time of the message

```math
x_{k+1}'=f(\mathbf{x}_k, \mathbf{u}_k) =
\begin{bmatrix}
x_{k+1} \\
y_{k+1} \\
\theta_{k+1} \\
v_{k+1}
\end{bmatrix}
=
\begin{bmatrix}
x_k + v_k \cos(\theta_k) \Delta t \\
y_k + v_k \sin(\theta_k) \Delta t \\
\theta_k + \omega^{gyro}_k \Delta t \\
v_k + a_k^{acc} \Delta t
\end{bmatrix}
```

GPS provides:

```math
\mathbf{z}_{k+1} =
\begin{bmatrix}
x_{k+1}^{GPS} \\
y_{k+1}^{GPS}
\end{bmatrix}
```


And the measurement function is:

```math
h(\mathbf{x}_{k+1}') =
\begin{bmatrix}
x_{k+1}' \\
y_{k+1}'
\end{bmatrix}
```

### Jacobians
```math
F_k =\frac{\partial f}{\partial \mathbf{x}}|_{\mathbf{x}=\mathbf{x}_k}= 
\begin{bmatrix}
1 & 0 & -v_k \sin(\theta_k) \Delta t & \cos(\theta_k) \Delta t \\
0 & 1 &  v_k \cos(\theta_k) \Delta t & \sin(\theta_k) \Delta t \\
0 & 0 & 1 & 0 \\\
0 & 0 & 0 & 1
\end{bmatrix}
```

```math
H_{k+1} =\frac{\partial h}{\partial \mathbf{x}}|_{\mathbf{x}=\mathbf{x}_{k+1}'} =
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0
\end{bmatrix}
```



#### Prediction Step

```math
\mathbf{x}_{k+1}' = f(\hat{\mathbf{x}}_k, \mathbf{u}_k)
```

```math 
P_{k+1}' = F_k P_k F_k^\top + Q
```

Update Step (if GPS available):

```math 
\mathbf{\hat{y}}_{k+1} = \mathbf{z}_{k+1} - h(\mathbf{x}_{k+1})
```

```math 
S_{k+1} = K_{k+1} P_{k+1}' H_{k+1}^T + R
```

```math
K_{k+1} = P_{k+1}' H_{k+1}^T S_{k+1}^{-1}
```

```math 
\mathbf{x}_{k+1} = \mathbf{x}_{k+1}' + K_{k+1} \mathbf{y}_{k+1}
```

```math
P_{k+1} = (I - K_{k+1} H_{k+1}) P_{k+1}'
```

## Controller
To follow the given variant of Lemniscate of Gerono

```math
\mathbf{p}^g(t) =
\begin{bmatrix}
x_k \\
y_k
\end{bmatrix}
=
\begin{bmatrix}
-2\sin l \cos l \\
\quad 2(\sin l + 1)
\end{bmatrix}
```
Where l is
```math
l = 
\begin{cases}
    \frac{\pi *t}{10} - \frac{1}{2}\pi,& \text{if } t < 20\\
    \frac{3}{2}\pi,& \text{otherwise}
\end{cases}
```

I designed a controller which tracks the rate of curvature of the path $\omega^p(t)$, the velocity of the path $v^p(t)$, and corrects for the tracking errors $x^e_k(t)$, $y^e_k(t)$ and $\theta^e_k(t)$. This is also where the time factor of the Lemniscate of Gerono is coupled to the ekf timestep k.

```math
\begin{aligned}
v^{des}_k &= v^p(t) + k_1 * x^e_k(t) \\
\omega^{des}_k &= \omega^p(t) + k_2 * y^e_k(t) + k_3 * sin(\theta^e_k(t))
\end{aligned}
```

The position tracking errors are simply
```math
\begin{aligned}
x^e_k(t) &= x^g(t) - x_k\\
y^e_k(t) &= y^g(t) - y_k
\end{aligned}
```


Then, to compute the angular tracking error, I first get the forward vector in global coordinates 
```math
\mathbf{p}_k^{forward}
=
\begin{bmatrix}
cos(\theta_k) \\
sin(\theta_k)
\end{bmatrix}
```
The forward vector in of the path at time t in global coordinates can be approximated as the difference between a small timestep forward and the current timestep. I set $t_{\epsilon}=1^{-4}$ 
```math
\mathbf{p}_k^{g,forward}(t)
=
\mathbf{p}^g(t+t_{\epsilon})-\mathbf{p}^g(t)
```
Which can be used to estimate the angular tracking error $\theta^e$
```math
\theta^e_k(t) = arctan(\frac{\mathbf{p}_k^{forward}\cdot \mathbf{p}_k^{g, forward}(t)}{det(\mathbf{p}_k^{forward}, \mathbf{p}_k^{g, forward}(t))})
```

The path velocity I get by taking the distance of the forward path vector divided by $t_{\epsilon}$
```math
v^p(t)=
\frac{|\mathbf{p}_k^{g,forward}(t)|}{t_{\epsilon}}
```
And the path angular velocity by calculating the angle between the previous path vector and the current one and diviving by t_{\epsilon}
```math
\omega^p(t)=
arctan(\frac{\mathbf{p}_k^{g,forward}(t-t_{\epsilon}) \cdot \mathbf{p}_k^{g,forward}(t)}{det(\mathbf{p}_k^{g,forward}(t-t_{\epsilon}), \mathbf{p}_k^{g,forward}(t))})/t_{\epsilon}
```


