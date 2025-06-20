from typing import Annotated, Literal, cast

import numpy as np
import numpy.typing as npt

from utils import dataclass


@dataclass
class StateVector:
    x: np.floating
    y: np.floating
    theta: np.floating
    vl: np.floating
    vr: np.floating
    al: np.floating
    ar: np.floating


@dataclass
class Measurement:
    a_x: np.floating
    a_y: np.floating
    omega_gyro: np.floating
    x_gps: np.floating
    y_gps: np.floating

    

    def __sub__(self, meas1, meas2):
        return Measurement(
            meas2.ax - meas1.ax,
            meas2.ay - meas1.ay,
            meas2.omega_gyro - meas1.omega_gyro,
            meas2.x_gps - meas1.x_gps,
            meas2.y_gps - meas1.y_gps,
        )

    def __post_init__(self, )



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
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            ax=ax_pred,
            ay=ay_pred,
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

        y_hat = z-z_pred
        S = H @ cov_prior @ H + self.R
        K = cov_prior @ H.T @ np.linalg.inv(S)
        state_new = state_prior + K @ y_hat
        self.cov = (np.eye(len(x_prior)) - K_current @ H_current) @ cov_prior
