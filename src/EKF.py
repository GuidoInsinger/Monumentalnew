from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

from .datatypes import GPSMeasurement, InertialMeasurement, StateVector


class EKF:
    def __init__(
        self,
        cov0: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        Q: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        R: Annotated[npt.NDArray[np.float64], Literal["M", "M"]],
    ) -> None:
        self.cov: Annotated[npt.NDArray[np.float64], Literal["N", "N"]] = cov0
        self.Q = Q
        self.R = R

    def predict(
        self, state: StateVector, inertial_measurement: InertialMeasurement, dt: float
    ) -> StateVector:
        x_pred = state.x + state.v * np.cos(state.theta) * dt
        y_pred = state.y + state.v * np.sin(state.theta) * dt
        theta_pred = state.theta + inertial_measurement.omega_gyro * dt
        v_pred = state.v + inertial_measurement.a_x * dt

        state_prior = StateVector(x=x_pred, y=y_pred, theta=theta_pred, v=v_pred)

        F = np.array(
            [
                [
                    1.0,
                    0.0,
                    -state.v * np.sin(state.theta) * dt,
                    np.cos(state.theta) * dt,
                ],
                [
                    0.0,
                    1.0,
                    state.v * np.cos(state.theta) * dt,
                    np.sin(state.theta) * dt,
                ],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.cov = F @ self.cov @ F.T + self.Q

        return state_prior

    def update(
        self,
        state_prior: StateVector,
        position_measurement: GPSMeasurement,
    ) -> StateVector:
        measurement_pred = np.array([state_prior.x, state_prior.y])

        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        y_hat = position_measurement.vec - measurement_pred
        cov_prior = np.copy(self.cov)

        S = H @ cov_prior @ H.T + self.R
        K = cov_prior @ H.T @ np.linalg.inv(S)

        state_new = state_prior + StateVector(*(K @ y_hat))
        self.cov = (np.eye(H.shape[1]) - K @ H) @ cov_prior

        return state_new
