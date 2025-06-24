from typing import cast

import numpy as np
import pandas as pd

from .controller import Controller
from .EKF import EKF
from .utils.datatypes import (
    GPSMeasurement,
    InertialMeasurement,
    RobotDimensions,
    StateVector,
)


class Robot:
    def __init__(
        self,
        dimensions: RobotDimensions,
        state0: StateVector,
        ekf: EKF,
        controller: Controller,
    ):
        self.dimensions = dimensions
        self.ekf = ekf
        self.controller = controller

        self.state_hist: list[StateVector] = [state0]
        self.inertial_hist: list[InertialMeasurement] = []
        self.gps_hist: list[GPSMeasurement] = []

    def process_gps(self, gps_data: list[float], gps_timestamp: pd.Timestamp) -> None:
        print("processing GPS")
        position_measurement = GPSMeasurement(
            x_gps=cast(np.floating, gps_data[0]),
            y_gps=cast(np.floating, gps_data[1]),
            timestamp=gps_timestamp,
        )
        state_updated = self.ekf.update(  # update state according to gps measurement
            state_prior=self.state_hist[-1],
            position_measurement=position_measurement,
        )
        # update histories
        self.gps_hist.append(position_measurement)
        self.state_hist.append(state_updated)

    def process_inertial(
        self,
        accelerometer_data: list[float],
        gyro_data: list[float],
        inertial_timestamp: pd.Timestamp,
    ) -> None:
        print("processing inertial")
        intertial_measurement = InertialMeasurement(
            a_x=cast(np.floating, accelerometer_data[0]),
            a_y=cast(np.floating, accelerometer_data[1]),
            omega_gyro=cast(np.floating, gyro_data[0]),
            timestamp=inertial_timestamp,
        )
        if len(self.inertial_hist) > 0:
            dt = (
                inertial_timestamp - self.inertial_hist[-1].timestamp
            ).total_seconds()  # assumes gps always arrives with an inertial measurement
        else:
            dt = 0.05  # Approximation for first measurement

        state_prior = self.ekf.predict(  # compute predicted next state
            state=self.state_hist[-1],
            inertial_measurement=intertial_measurement,
            dt=dt,
        )

        self.state_hist.append(state_prior)
        self.inertial_hist.append(intertial_measurement)
