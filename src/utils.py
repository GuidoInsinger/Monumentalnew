from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional

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

    def __add__(self, meas2: "StateVector"):
        return StateVector(*(self.vec + meas2.vec))


@dataclass
class Measurement:
    a_x: np.floating
    a_y: np.floating
    omega_gyro: np.floating
    x_gps: Optional[np.floating]
    y_gps: Optional[np.floating]

    vec: Annotated[npt.NDArray[np.floating], Literal["M"]] = field(init=False)

    def __post_init__(self):
        if (self.x_gps is None) and (self.y_gps is None):
            self.vec = np.array(
                [
                    self.a_x,
                    self.a_y,
                    self.omega_gyro,
                ]
            )
        else:
            try:
                self.vec = np.array(
                    [self.a_x, self.a_y, self.omega_gyro, self.x_gps, self.y_gps]
                )
            except Exception as ex:
                print(ex)

    def __len__(self):
        return len(self.vec)

    def __sub__(self, meas2: "Measurement"):
        if len(self) != len(meas2):
            raise Exception("Cannot subtract measurements of differing size")

        if (self.x_gps is not None) and (self.y_gps is not None):
            return Measurement(*(self.vec - meas2.vec))

        else:
            return Measurement(
                a_x=self.a_x - meas2.a_x,
                a_y=self.a_y - meas2.a_y,
                omega_gyro=self.omega_gyro - meas2.omega_gyro,
                x_gps=None,
                y_gps=None,
            )


@dataclass
class Controls:
    v_l_desired: np.floating
    v_r_desired: np.floating


@dataclass
class Robot:
    L: float
    epsilon: float
