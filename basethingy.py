from typing import Annotated, Literal

import numpy as np
from numpy.f2py.auxfuncs import containsderivedtypes
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

@dataclass
class Controls:
    v_l_desired: np.floating
    v_r_desired: np.floating

@dataclass
class Robot:
    L: np.floating

class EKF:
    def __init__(
        self,
        robot: Robot,

        P0: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],

        Q: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
        R: Annotated[npt.NDArray[np.float64], Literal["M", "M"]],

    ) -> None:
        self.P = P0
        self.Q = Q
        self.R = R

    def predict(
        self, state:StateVector, controls: Controls, dt: float
    ) -> 
        x_pred = state.x + (state.vl+state.vr)*np.cos(state.theta)/2*dt
        y_pred = state.y + (state.vl+state.vr)*np.cos(state.theta)/2*dt
        theta = state.theta + (-state.vl+state.vr)/self.robot.L*dt
        vl = state.vl + state.al*dt
        vr = state.vr + state.ar*dt
        al = input.v

        x_prior = self.f(u=u, dt=dt)
        cov_prior = F_current @ self.P @ F_current.T + Q_current

        return x_prior, cov_prior

    def update(
        self,
        z: Annotated[npt.NDArray[np.float64], Literal["M"]],
        x_prior: Annotated[npt.NDArray[np.float64], Literal["N"]],
        cov_prior: Annotated[npt.NDArray[np.float64], Literal["N", "N"]],
    ):
        y_hat = z - self.h(x_prior=x_prior)  #
        H_current = self.H(x_prior=x_prior)
        R_current = self.R()

        S_current = H_current @ cov_prior @ H_current.T + R_current
        K_current = cov_prior @ H_current.T @ np.linalg.inv(S_current)
        self.x = x_prior + K_current @ y_hat
        self.cov = (np.eye(len(x_prior)) - K_current @ H_current) @ cov_prior
