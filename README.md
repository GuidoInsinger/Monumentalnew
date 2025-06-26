# Differential drive control

Welcome to my solution for the Momumental controls assignment!

## Installation

* First install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it yet
* Then clone this repository
```
git clone https://github.com/GuidoInsinger/Monumentalnew.git
```
* Navigate to the folder
```
cd Monumentalnew
```
* Install the dependencies using `uv sync`
```
uv sync
```
* Run the code using `uv run main.py`
```
uv run main.py
```

## Visualization

In my visualization I built a 3d view that shows all of the important components
* The purple path is the path to be followed, with a purple arrow indicating the current desired position
* The yellow dots are the recieved GPS measurements and the yellow arrows coming from the robot are the receieved acceleration vectors (these are sometimes occluded by the robot body)
* The pink path is the history of the EKF estimate. This path will be a bit jagged due to the GPS corrections coming in
* On the right you can see the history of the individual states estimated by the EKF 

https://github.com/user-attachments/assets/4b7726e6-df3f-48be-8b4b-7380455f19c9

## Results

My solution seems to track the input path quite well. Sometimes if GPS measurements are far from the predicted position, it takes a second to get back on track but that can be expected from noisy measurements. The score I get is usually around 1-1.5, depending on the specific run.

## Approach

To estimate the state of the robot and deal with the asynchronous measurements I chose an EKF. In my EKF I use the inertial measurements as inputs to the prediction step, and the GPS measurements as inputs for the update step. This allows me to run the prediction step whenever a new inertial measurement comes in which I found to be around ~20 hz and an update step to correct the predictions using the GPS measurement at around ~1hz. The uncertainty propagation ensures the EKF covariance will correctly encapsulate the compounded uncertainies from the ~20 prediction steps happening up until the point a GPS measurement comes in. 

Another reason for why I chose to use inertial measurements in my update step rather than the true inputs is that I found the actuator dynamics to be quite noisy and with no explicit noise model given. Setting up a proper model to track this seemed more complicated than a simple kinematic estimator, and I doubt that it would be much more performant than this solution. 

### EKF state estimation

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

Discretized equations of motion with Euler integration over timestep $\Delta t$, where I get the value of $\Delta t$ from the arrival time of the message:

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

#### Jacobians
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

#### Update Step (if GPS available):

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

#### Q and R

To get an estimate of the uncertainties of the sensor data I sent zero input controls to the server for ~60 seconds and calculated the variance in each signal. With this method I found

```math
\begin{aligned}
\sigma^2_{a^{acc}} \approx 0.0004\\
\sigma^2_{\omega^{gyro}} \approx 0.0004\\
\sigma^2_{x^{GPS}} \approx 0.1\\
\sigma^2_{y^{GPS}} \approx 0.1
\end{aligned}
```

I initially chose Q such that the variances on $\omega$ and $v$ align with these estimated uncertainties, and I set the position variances to be the same as on the v state. This second part is not necessarily true because the uncertainties on x and y are dependent on the over time integrated error, but seems to work fine.

After some experimentation I ended up dividing all values of Q by 10 which leads to more stable results, which is probably due to my controller being tuned quite aggresively and performing better when the state estimate doesn't jump around as much.

```math
Q=
\Delta t
\begin{bmatrix}
\sigma^2_{a^{acc}} & 0 & 0 & 0 \\
0 & \sigma^2_{a^{acc}} & 0 & 0\\
0 & 0 & \sigma^2_{\omega^{gyro}} & 0 \\\
0 & 0 & 0 & \sigma^2_{a^{acc}}
\end{bmatrix}
```

I choose R to align with my estimated variance on $x^{GPS}$ and $y^{GPS}$

```math
R=
\begin{bmatrix}
\sigma^2_{x^{GPS}} & 0 \\
0 & \sigma^2_{y^{GPS}}
\end{bmatrix}
```
#### Initial conditions

The initial state is as per the assignment
```math
\mathbf{x}_0 = 
\begin{bmatrix}
x_0 \\
y_0 \\
\theta_0 \\
v_0
\end{bmatrix}
=
\begin{bmatrix}
0.0 \\
0.0 \\
0.0 \\
0.0 
\end{bmatrix}
```

I simply set the initial covariance to some small values, given that the initial state is certain to be zeros.

```math
P_0
=
\begin{bmatrix}
1^{-6} & 0 & 0 & 0 \\
0 & 1^{-6} & 0 & 0\\
0 & 0 & 1^{-7} & 0 \\\
0 & 0 & 0 & 1^{-6}
\end{bmatrix}
```

### Controller

To follow the given variant of Lemniscate of Gerono

```math
\mathbf{p}^g(t) =
\begin{bmatrix}
x^g_k(t) \\
y^g_k(t)
\end{bmatrix}
=
\begin{bmatrix}
-2\sin l(t) \cos l(t) \\
\quad 2(\sin l(t) + 1)
\end{bmatrix}
```
Where l is
```math
l(t) = 
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

In my final solution I used the parameters
```math
\begin{aligned}
k_1 &= 2.5 \\
k_2 &= -8.0 \\
k_3 &= -12.0
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
Which can be used to estimate the angular tracking error $\theta^e_k(t)$ at time t 
```math
\theta^e_k(t) = arctan(\frac{\mathbf{p}_k^{forward}\cdot \mathbf{p}_k^{g, forward}(t)}{det(\mathbf{p}_k^{forward}, \mathbf{p}_k^{g, forward}(t))})
```

The path velocity I get by taking the distance of the forward path vector divided by $t_{\epsilon}$
```math
v^p(t)=
\frac{|\mathbf{p}_k^{g,forward}(t)|}{t_{\epsilon}}
```
And the path angular velocity by calculating the angle between the previous path vector and the current one and diviving by $t_{\epsilon}$
```math
\omega^p(t)=
arctan(\frac{\mathbf{p}_k^{g,forward}(t-t_{\epsilon}) \cdot \mathbf{p}_k^{g,forward}(t)}{det(\mathbf{p}_k^{g,forward}(t-t_{\epsilon}), \mathbf{p}_k^{g,forward}(t))})/t_{\epsilon}
```

The final control inputs that are sent back to the server are computed as follows
```math
\begin{aligned}
v^{left}_k &= v^{des}_k + \omega^{des}_k \frac{L}{2}\\
v^{right}_k &= v^{des}_k - \omega^{des}_k \frac{L}{2}
\end{aligned}
```

## Possible improvements
* I tuned the controller by just looking at the behaviour, this could be further improved
* Including the second accelerometer measurement could potentially improve the state estimation
* Using a dynamical model of a differential drive robot could improve the tracking performance by reducing the dynamics uncertainty. To keep using an EKF would have to incorporate a Time-Variant measurement model that uses the different measurements available at each timestep.
* More experimentation with the specific values of Q could possibly improve performance
