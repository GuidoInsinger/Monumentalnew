import json
import time
from typing import cast

import numpy as np
import pandas as pd
import websocket

from src.EKF import EKF
from src.utils import Controls, GPSMeasurement, InertialMeasurement, StateVector, gerono
from src.viz import init_rr, update_rr


class VizualizeEKF(websocket.WebSocketApp):
    def __init__(self):
        super().__init__(  # type:ignore
            "ws://91.99.103.188:8765",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        state0 = StateVector(*np.array([0.0, 0.0, 0.0, 0.0]))

        cov0 = np.diag([1e-4, 1e-4, 1e-5, 1e-4])

        Q = 0.05 * np.diag([4e-3, 4e-3, 4e-3, 4e-3])
        R = np.diag([0.1, 0.1])

        self.ekf = EKF(cov0=cov0, Q=Q, R=R)
        self.start_time = time.time()

        self.state_hist: list[StateVector] = [state0]

        self.inertial_hist: list[InertialMeasurement] = []
        self.gps_hist: list[GPSMeasurement] = []

        init_rr()

    def on_message(self, ws: websocket.WebSocket, message: str):
        t = time.time() - self.start_time
        msg = json.loads(message)
        if msg["message_type"] == "score":
            print(msg)
            self.close()  # type:ignore

        elif msg["message_type"] == "sensors":
            sensors = msg["sensors"]

            acc_data = sensors[0]["data"]
            gyro_data = sensors[1]["data"]
            gps_data = sensors[2]["data"]

            inertial_timestamp = pd.Timestamp(
                sensors[0]["timestamp"]
            )  # assumes gyro_timestamp is equal

            gps_timestamp = pd.Timestamp(sensors[2]["timestamp"])

            intertial_measurement = InertialMeasurement(
                a_x=cast(np.floating, acc_data[0]),
                a_y=cast(np.floating, acc_data[1]),
                omega_gyro=cast(np.floating, gyro_data[0]),
                timestamp=inertial_timestamp,
            )
            # do update first, this works because the server never sends a gps measurement first
            # and is necessary such that the prediction for the right timestep is used

            if len(self.gps_hist) > 0:
                if self.gps_hist[-1].timestamp != gps_timestamp:
                    position_measurement = GPSMeasurement(
                        x_gps=gps_data[0], y_gps=gps_data[1], timestamp=gps_timestamp
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
                    x_gps=gps_data[0], y_gps=gps_data[1], timestamp=gps_timestamp
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

            # print("inertial update")

            t_epsilon = 1e-3
            path_dir = gerono(t=t + t_epsilon) - gerono(t=t)
            path_velocity = np.linalg.norm(path_dir / t_epsilon)

            path_omega = -np.acos(
                np.dot(
                    path_dir,
                    np.array(
                        [
                            np.cos(self.state_hist[-1].theta),
                            np.sin(self.state_hist[-1].theta),
                        ]
                    ),
                )
                / np.linalg.norm(path_dir)
            )
            path_angle = -np.acos(
                np.dot(
                    path_dir,
                    np.array(
                        [
                            np.cos(self.state_hist[-1].theta),
                            np.sin(self.state_hist[-1].theta),
                        ]
                    ),
                )
                / np.linalg.norm(path_dir)
            )
            # print(path_angle)
            pos_error = gerono(t=t) - self.state_hist[-1].vec[:2]
            x_e, y_e = (
                np.array(
                    [
                        [
                            np.cos(self.state_hist[-1].theta),
                            np.sin(self.state_hist[-1].theta),
                        ],
                        [
                            -np.sin(self.state_hist[-1].theta),
                            np.cos(self.state_hist[-1].theta),
                        ],
                    ]
                )
                @ pos_error
            )

            k1 = 2.5
            k2 = 1.5
            k3 = 5.0

            v_des = path_velocity * np.cos(path_angle) + k1 * x_e
            omega_des = ky * y_e + ktheta * path_angle
            print(path_angle)

            controls = Controls(
                *np.array([v_des + omega_des * 0.25, v_des - omega_des * 0.25])
            )
            inputs = {
                "v_left": controls.v_l_desired,
                "v_right": controls.v_r_desired,
            }
            update_rr(self.state_hist, t=t, gps_hist=self.gps_hist)

            ws.send(json.dumps(inputs))

    def on_error(self, ws: websocket.WebSocket, error: Exception):
        print(error)

    def on_close(self, ws: websocket.WebSocket, close_status_code: int, close_msg: str):
        print("### closed ###")

    def on_open(self, ws: websocket.WebSocket):
        print("Opened connection")


if __name__ == "__main__":
    # websocket.enableTrace(True)
    viz_ekf = VizualizeEKF()

    # ws = websocket.WebSocketApp(
    #     "ws://91.99.103.188:8765",
    #     on_open=viz_ekf.on_open,
    #     on_message=viz_ekf.on_message,
    #     on_error=viz_ekf.on_error,
    #     on_close=viz_ekf.on_close,
    # )

    viz_ekf.run_forever()  # type: ignore
