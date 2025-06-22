from dataclasses import dataclass, field
from typing import Annotated, Literal

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
        self.vl = np.clip(self.vl, -2.0, 2.0)
        self.vr = np.clip(self.vr, -2.0, 2.0)

        self.al = np.clip(self.al, -1.0, 1.0)
        self.ar = np.clip(self.ar, -1.0, 1.0)

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
        print(" hi")

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
