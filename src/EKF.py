from typing import Annotated, Literal, cast

import numpy as np
import numpy.typing as npt

from .utils import Controls, Measurement, Robot, StateVector


class EKF:
    def __init__(
        self,
        robot: Robot,
        cov0: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        Q: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        R: Annotated[npt.NDArray[np.float64], Literal["M", "M"]],
    ) -> None:
        self.robot = robot
        self.cov = cov0
        self.Q = Q
        self.R = R

    def predict(
        self, state: StateVector, controls: Controls, dt: float
    ) -> tuple[StateVector, Annotated[npt.NDArray[np.float64], Literal["N", "N"]]]:
        x_pred = state.x + (state.vl + state.vr) * np.cos(state.theta) / 2 * dt
        y_pred = state.y + (state.vl + state.vr) * np.sin(state.theta) / 2 * dt
        theta_pred = state.theta + (-state.vl + state.vr) / self.robot.L * dt

        vl_pred = np.clip(state.vl + state.al * dt, -2.0, 2.0)
        vr_pred = np.clip(state.vr + state.ar * dt, -2.0, 2.0)

        delta_vl = controls.v_l_desired - state.vl
        delta_vr = controls.v_r_desired - state.vr

        al_pred = cast(
            np.floating,
            0.5 * np.sign(delta_vl) if delta_vl > self.robot.epsilon else 0.0,
        )
        ar_pred = cast(
            np.floating,
            0.5 * np.sign(delta_vr) if delta_vr > self.robot.epsilon else 0.0,
        )

        x_prior = StateVector(
            x=x_pred,
            y=y_pred,
            theta=theta_pred,
            vl=vl_pred,
            vr=vr_pred,
            al=al_pred,
            ar=ar_pred,
        )

        F = np.array(
            [
                [
                    1.0,
                    0.0,
                    -(state.vl + state.vr) / 2 * np.sin(state.theta),
                    np.cos(state.theta) * dt / 2,
                    np.cos(state.theta) * dt / 2,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    (state.vl + state.vr) / 2 * np.cos(state.theta),
                    np.sin(state.theta) * dt / 2,
                    np.sin(state.theta) * dt / 2,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 1.0, -dt / self.robot.L, dt / self.robot.L, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        cov_prior = F @ self.cov @ F.T + self.Q

        return x_prior, cov_prior

    def update(
        self,
        state_prior: StateVector,
        cov_prior: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        z: Measurement,
    ):
        ax_pred = (state_prior.al + state_prior.ar) / 2
        ay_pred = (state_prior.vr**2 - state_prior.vl**2) / (2 * self.robot.L)
        omega_gyro_pred = (state_prior.vr - state_prior.vl) / self.robot.L
        x_gps_pred = state_prior.x
        y_gps_pred = state_prior.y

        z_pred = Measurement(
            a_x=ax_pred,
            a_y=ay_pred,
            omega_gyro=omega_gyro_pred,
            x_gps=x_gps_pred,
            y_gps=y_gps_pred,
        )

        H = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                [
                    0.0,
                    0.0,
                    0.0,
                    -state_prior.vl / self.robot.L,
                    state_prior.vr / self.robot.L,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -1 / self.robot.L,
                    1 / self.robot.L,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )

        if len(z) == 3:
            z_pred = Measurement(
                a_x=z_pred.a_x,
                a_y=z_pred.a_y,
                omega_gyro=z_pred.omega_gyro,
                x_gps=None,
                y_gps=None,
            )
            H = H[:3]  # type:ignore

        print(z.vec, z_pred.vec)
        y_hat = z - z_pred
        S = H @ cov_prior @ H.T + self.R[: len(z), : len(z)]
        K = cov_prior @ H.T @ np.linalg.inv(S)

        state_new = state_prior + StateVector(*(K @ y_hat.vec))
        self.cov = (np.eye(H.shape[1]) - K @ H) @ cov_prior

        return state_new


def gerono(t: np.floating) -> tuple[float, float]:
    if t < 20:
        k = np.pi * t / 10 - np.pi / 2
    else:
        k = 3 * np.pi / 2

    x = -2 * np.sin(k) * np.cos(k)
    y = 2 * (np.sin(k) + 1)

    return x, y


if __name__ == "__main__":
    import time

    from viz import init_rr, update_rr

    robot = Robot(L=0.5, epsilon=2.0)
    cov0 = 0.1 * np.eye(7)

    Q = np.diag([0.1, 0.1, 0.01, 0.3, 0.3, 3, 3])
    R = np.diag([0.1, 0.1, 0.1, 0.05, 0.05])
    ekf = EKF(robot=robot, cov0=cov0, Q=Q, R=R)

    state0 = StateVector(*np.random.normal(loc=0.3, scale=0.1, size=(7)))
    u0 = Controls(*np.random.normal(loc=0.3, scale=0.1, size=(2)))

    start = time.time()

    statehist: list[StateVector] = [state0]
    init_rr()

    while time.time() - start < 10:
        state_prior, cov_prior = ekf.predict(state=statehist[-1], controls=u0, dt=0.1)
        z = Measurement(*np.random.normal(loc=0.3, scale=2.5, size=(5)))
        state_new = ekf.update(state_prior=state_prior, cov_prior=cov_prior, z=z)
        statehist.append(state_new)
        update_rr(statehist, t=time.time() - start)
        time.sleep(0.1)
