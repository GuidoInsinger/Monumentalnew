from typing import cast

import numpy as np
import pandas as pd

from .controller import Controller
from .datatypes import (
    GPSMeasurement,
    InertialMeasurement,
    RobotDimensions,
    SensorMessage,
    StateVector,
)
from .EKF import EKF


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

    def process_sensordata(self, sensormessages: list[SensorMessage]):
        accelerometer_message = sensormessages[0]
        gyro_message = sensormessages[1]
        gps_message = sensormessages[2]

        accelerometer_data = accelerometer_message["data"]
        gyro_data = gyro_message["data"]
        gps_data = gps_message["data"]

        inertial_timestamp = pd.Timestamp(
            accelerometer_message["timestamp"]
        )  # assumes gyro_timestamp is equal to accelerometer timestamp

        gps_timestamp = pd.Timestamp(gps_message["timestamp"])

        intertial_measurement = InertialMeasurement(
            a_x=cast(np.floating, accelerometer_data[0]),
            a_y=cast(np.floating, accelerometer_data[1]),
            omega_gyro=cast(np.floating, gyro_data[0]),
            timestamp=inertial_timestamp,
        )
        # do update first, this works because the server never sends a gps measurement first
        # and is necessary such that the prediction for the right timestep is used

        if len(self.gps_hist) > 0:
            if self.gps_hist[-1].timestamp != gps_timestamp:
                position_measurement = GPSMeasurement(
                    x_gps=cast(np.floating, gps_data[0]),
                    y_gps=cast(np.floating, gps_data[1]),
                    timestamp=gps_timestamp,
                )

                self.gps_hist.append(position_measurement)

                state_updated = self.ekf.update(
                    state_prior=self.state_hist[-1],
                    position_measurement=position_measurement,
                )
                self.state_hist.append(state_updated)
                # print("gps update")
        else:
            position_measurement = GPSMeasurement(
                x_gps=cast(np.floating, gps_data[0]),
                y_gps=cast(np.floating, gps_data[1]),
                timestamp=gps_timestamp,
            )
            self.gps_hist.append(position_measurement)

            state_updated = self.ekf.update(
                state_prior=self.state_hist[-2],
                position_measurement=position_measurement,
            )
            self.state_hist.append(state_updated)

        if len(self.inertial_hist) > 0:
            dt = (
                inertial_timestamp - self.inertial_hist[-1].timestamp
            ).total_seconds()  # assumes gps always arrives with an inertial measurement
        else:
            dt = 0.05  # roughly true

        state_prior = self.ekf.predict(
            state=self.state_hist[-1],
            inertial_measurement=intertial_measurement,
            dt=dt,
        )

        self.state_hist.append(state_prior)
        self.inertial_hist.append(intertial_measurement)
