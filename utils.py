from abc import ABC
from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, cast

import numpy as np
import numpy.typing as npt


@dataclass
class StateVec:
    """
    Statevector of length N
    """

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


class StateCov:
    cov: Annotated[npt.NDArray[np.floating], Literal["N"]] = field(init=False)


@dataclass
class Measurement:
    """
    Measurement of length M

    If no new GPS is available
    """

    a_x: float
    a_y: float
    omega_gyro: float
    x_gps: Optional[float]
    y_gps: Optional[float]

    # dt: float

    # cov: Annotated[npt.NDArray[np.float64], Literal["M", "M"]]


@dataclass
class Controls:
    vl_setpoint: float
    vr_setpoint: float


@dataclass
class DiffDriveRobot:
    L: float
    epsilon: float


class EKF(ABC):
    def __init__(
        self,
        robot: DiffDriveRobot,
        Q: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        R: Annotated[npt.NDArray[np.float64], Literal["M", "M"]],
        cov0: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
    ) -> None:
        self.robot = robot
        self.Q = Q
        self.R = R
        self.cov: Annotated[npt.NDArray[np.float64], Literal["N", "N"]] = cov0

    def predict(
        self, state: StateVec, controls: Controls, dt: float
    ) -> tuple[StateVec, Annotated[npt.NDArray[np.float64], Literal["N", "N"]]]:
        # calculate state derivatives
        xdot = (state.vr + state.vl) / 2 * np.cos(state.theta)
        ydot = (state.vr + state.vl) / 2 * np.sin(state.theta)
        thetadot = (state.vr - state.vl) / self.robot.L

        delta_vl = controls.vl_setpoint - state.vl
        delta_vr = controls.vr_setpoint - state.vr

        al = cast(
            np.floating,
            0.5 * np.sign(delta_vl) if delta_vl > self.robot.epsilon else 0.0,
        )
        ar = cast(
            np.floating,
            0.5 * np.sign(delta_vr) if delta_vr > self.robot.epsilon else 0.0,
        )

        vldot = al
        vrdot = ar

        # calculate state prior
        x_new = state.x + xdot * dt
        y_new = state.y + ydot * dt

        theta_new = state.theta + thetadot * dt

        vr_new = state.vr + vrdot * dt
        vl_new = state.vl + vldot * dt

        ar_new = ar
        al_new = al

        # calculate Jacobian
        F = np.array(
            [
                [
                    1.0,
                    0.0,
                    -(state.vr + state.vl) / 2 * np.sin(state.theta) * dt,
                    np.cos(state.theta) * dt / 2,
                    np.cos(state.theta) * dt / 2,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    (state.vr + state.vl) / 2 * np.cos(state.theta) * dt,
                    np.cos(state.theta) * dt / 2,
                    np.cos(state.theta) * dt / 2,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    dt / 2,
                    -dt / 2,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        # calculate covariance prior
        cov_prior = F @ self.cov @ F.T + self.Q

        return StateVec(
            x_new, y_new, theta_new, vl_new, vr_new, al_new, ar_new
        ), cov_prior

    def update(
        self,
        z: Measurement,
        state_prior: StateVec,
        cov_prior: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
    ):
        # predict outcomes
        a_x_pred = (state_prior.al + state_prior.ar) / 2
        a_y_pred = (state_prior.vr**2 - state_prior.vl**2) / (2 * self.robot.L)
        omega_gyro_pred = (state_prior.vr - state_prior.vl) / self.robot.L
        x_gps_pred = state_prior.x
        y_gps_pred = state_prior.y

        measurement_predict = Measurement(
            a_x=a_x_pred,
            a_y=a_y_pred,
            omega_gyro=omega_gyro_pred,
            x_gps=x_gps_pred,
            y_gps=y_gps_pred,
        )

        y_hat = z - self.h(state_prior=state_prior)  #
        H_current = self.H(state_prior=state_prior)
        R_current = self.R()

        S_current = H_current @ cov_prior @ H_current.T + R_current
        K_current = cov_prior @ H_current.T @ np.linalg.inv(S_current)
        self.x = state_prior + K_current @ y_hat
        self.cov = (np.eye(len(state_prior)) - K_current @ H_current) @ cov_prior
