from dataclasses import dataclass, field
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class StateVector:
    x: np.floating
    y: np.floating
    theta: np.floating
    v: np.floating

    vec: Annotated[npt.NDArray[np.floating], Literal["N"]] = field(init=False)

    def __post_init__(self):
        self.v = np.clip(self.v, -2.0, 2.0)

        if self.theta > np.pi:
            self.theta -= 2 * np.pi
        elif self.theta < -np.pi:
            self.theta += 2 * np.pi

        self.vec = np.array([self.x, self.y, self.theta, self.v])

    def __add__(self, meas2: "StateVector"):
        return StateVector(*(self.vec + meas2.vec))


@dataclass
class InertialMeasurement:
    a_x: np.floating
    a_y: np.floating
    omega_gyro: np.floating

    timestamp: pd.Timestamp


@dataclass
class GPSMeasurement:
    x_gps: np.floating
    y_gps: np.floating

    timestamp: pd.Timestamp

    vec: Annotated[npt.NDArray[np.floating], Literal[2]] = field(init=False)

    def __post_init__(self):
        self.vec = np.array([self.x_gps, self.y_gps])


@dataclass
class Controls:
    v_l_desired: np.floating
    v_r_desired: np.floating

    def __post_init__(self):
        self.v_l_desired = np.clip(self.v_l_desired, -2.0, 2.0)
        self.v_r_desired = np.clip(self.v_r_desired, -2.0, 2.0)


def gerono(t: float) -> npt.NDArray[np.floating]:
    if t < 20:
        k = np.pi * t / 10 - np.pi / 2
    else:
        k = 3 * np.pi / 2

    x = -2 * np.sin(k) * np.cos(k)
    y = 2 * (np.sin(k) + 1)

    return np.array([x, y])


def angle_between_vectors(
    vec1: npt.NDArray[np.floating], vec2: npt.NDArray[np.floating]
):
    """
    Will return the counterclockwise angle from vec1 to vec2
    """

    dot = np.dot(vec1, vec2)
    det = np.linalg.det(np.vstack((vec1, vec2)))

    return np.atan2(det, dot)
    # vec1_u = vec1 / np.linalg.norm(vec1)
    # vec2_u = vec2 / np.linalg.norm(vec2)

    # return np.arccos(np.dot(vec1_u, vec2_u))


print(angle_between_vectors(np.array([1, 0]), np.array([1, 0])))
