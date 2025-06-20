from dataclasses import dataclass, field
from typing import Annotated, Literal, cast

import numpy as np
import numpy.typing as npt


@dataclass
class StateVector:
    x: np.floating
    y: np.floating
    theta: np.floating
    vl: np.floating
    vr: np.floating
    al: np.floating
    ar: np.floating

    vec: Annotated[npt.NDArray[np.floating], Literal["N"]] = field(init=False)

    def __post_init__(self):
        self.vec = np.array(
            [self.x, self.y, self.theta, self.vl, self.vr, self.al, self.ar]
        )

    def __add__(self, meas2):
        return StateVector(*(self.vec - meas2.vec))


@dataclass
class Measurement:
    a_x: np.floating
    a_y: np.floating
    omega_gyro: np.floating
    x_gps: np.floating
    y_gps: np.floating

    vec: Annotated[npt.NDArray[np.floating], Literal["M"]] = field(init=False)

    def __post_init__(self):
        self.vec = np.array(
            [self.a_x, self.a_y, self.omega_gyro, self.x_gps, self.y_gps]
        )

    def __sub__(self, meas2):
        return Measurement(*(self.vec - meas2.vec))


@dataclass
class Controls:
    v_l_desired: np.floating
    v_r_desired: np.floating


@dataclass
class Robot:
    L: np.floating
    epsilon: np.floating


class EKF:
    def __init__(
        self,
        robot: Robot,
        P0: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        Q: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        R: Annotated[npt.NDArray[np.float64], Literal["M", "M"]],
    ) -> None:
        self.robot = robot
        self.P = P0
        self.Q = Q
        self.R = R

    def predict(
        self, state: StateVector, controls: Controls, dt: float
    ) -> tuple[StateVector, Annotated[npt.NDArray[np.float64], Literal["N", "N"]]]:
        x_pred = state.x + (state.vl + state.vr) * np.cos(state.theta) / 2 * dt
        y_pred = state.y + (state.vl + state.vr) * np.cos(state.theta) / 2 * dt
        theta_pred = state.theta + (-state.vl + state.vr) / self.robot.L * dt
        vl_pred = state.vl + state.al * dt
        vr_pred = state.vr + state.ar * dt

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
                [0.0, 0.0, 0.0, 1.0, 0.0, state.al * dt, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, state.ar * dt],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        cov_prior = F @ self.P @ F.T + self.Q

        return x_prior, cov_prior

    def update(
        self,
        state_prior: StateVector,
        cov_prior: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        z: Measurement,
    ):
        ax_pred = (state_prior.al + state_prior.ar) / 2
        ay_pred = (state_prior.vr**2 - state_prior.vl**2) / 2 * self.robot.L
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

        y_hat = z - z_pred
        S = H @ cov_prior @ H.T + self.R
        K = cov_prior @ H.T @ np.linalg.inv(S)

        state_new = state_prior + StateVector(*(K @ y_hat.vec))
        self.P = (np.eye(len(state_new)) - K @ H.T) @ cov_prior

        return state_new


if __name__ == "__main__":
    robot = Robot(L=0.5, epsilon=0.1)
    P0 = np.random.normal(loc=0.3, scale=2.5, size=(7, 7))
    Q = np.random.normal(loc=0.3, scale=2.5, size=(7, 7))
    R = np.random.normal(loc=0.3, scale=2.5, size=(5, 5))
    ekf = EKF(robot=robot, P0=P0, Q=Q, R=R)

    x0 = StateVector(*np.random.normal(loc=0.3, scale=2.5, size=(7)))
    u = Controls(*np.random.normal(loc=0.3, scale=2.5, size=(2)))

    state_prior, cov_prior = ekf.predict(state=x0, controls=u, dt=0.1)

    z = Measurement(*np.random.normal(loc=0.3, scale=2.5, size=(5)))
    x_new = ekf.update(state_prior=state_prior, cov_prior=cov_prior, z=z)
