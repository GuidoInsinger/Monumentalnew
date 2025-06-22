import json
import time
from typing import cast

import numpy as np
import websocket

from src.EKF import EKF
from src.utils import Controls, Measurement, Robot, StateVector
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

        state0 = StateVector(*np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        u0 = Controls(*np.array([0.0, 0.0]))
        robot = Robot(L=0.5, epsilon=0.1)

        cov0 = np.diag([0.1, 0.1, 0.01, 0.3, 0.3, 0.5, 0.5])

        Q = np.diag([0.1, 0.1, 0.01, 0.3, 0.3, 0.5, 0.5])
        R = np.diag([0.01, 0.01, 0.01, 0.05, 0.05])

        self.ekf = EKF(robot=robot, cov0=cov0, Q=Q, R=R)
        self.start_time = time.time()

        self.state_hist: list[StateVector] = [state0]
        self.controls_hist: list[Controls] = [u0]
        self.measurement_hist: list[Measurement] = []
        init_rr()

    def update_ekf(self, z: Measurement, dt: float):
        state_prior, cov_prior = self.ekf.predict(
            state=self.state_hist[-1], controls=self.controls_hist[-1], dt=dt
        )
        state_new = self.ekf.update(state_prior=state_prior, cov_prior=cov_prior, z=z)
        self.state_hist.append(state_new)

    def parse_sensor_message(self, message: str):
        msg = json.loads(message)

        sensors = msg["sensors"]

        accdata = sensors[0]["data"]
        gyrodata = sensors[1]["data"]
        gpsdata = sensors[2]["data"]

        z = Measurement(
            a_x=cast(np.floating, accdata[0]),
            a_y=cast(np.floating, accdata[1]),
            omega_gyro=cast(np.floating, gyrodata[0]),
            x_gps=cast(np.floating, gpsdata[0]),
            y_gps=cast(np.floating, gpsdata[1]),
        )

        if len(self.measurement_hist) > 0:  # check for first measurement
            last_measurement = self.measurement_hist[-1]
            if (gpsdata[0] == last_measurement.x_gps) and (
                gpsdata[1] == last_measurement.y_gps
            ):
                # no new gps, overwrite to omit gps measurements
                z = Measurement(
                    a_x=cast(np.floating, accdata[0]),
                    a_y=cast(np.floating, accdata[1]),
                    omega_gyro=cast(np.floating, gyrodata[0]),
                    x_gps=None,
                    y_gps=None,
                )
        self.measurement_hist.append(z)
        return z

    def on_message(self, ws: websocket.WebSocket, message: str):
        msg = json.loads(message)
        if msg["message_type"] == "score":
            print(msg)

        elif msg["message_type"] == "sensors":
            z = self.parse_sensor_message(message=message)

            self.update_ekf(z=z, dt=0.05)  # TODO: update dt to align with true dt

            controls = Controls(*np.array([1.0, 1.0]))  # TODO: implement controller
            inputs = {
                "v_left": controls.v_l_desired,
                "v_right": controls.v_r_desired,
            }
            update_rr(self.state_hist, t=time.time() - self.start_time)
            self.controls_hist.append(controls)

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
