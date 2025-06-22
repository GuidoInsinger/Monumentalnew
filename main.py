import json
import time

import numpy as np
import websocket

from EKF import EKF
from utils import Controls, Measurement, Robot, StateVector
from viz import init_rr, update_rr


class VizualizeEKF(websocket.WebSocketApp):
    def __init__(self):
        super().__init__(
            "ws://91.99.103.188:8765",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        state0 = StateVector(*np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0]))

        robot = Robot(L=0.5, epsilon=1.0)
        P0 = 0.1 * np.random.normal(loc=0.3, scale=2.5, size=(7, 7))
        Q = np.diag([0.1, 0.1, 0.01, 0.3, 0.3, 0.5, 0.5])
        R = np.diag([0.1, 0.1, 0.1, 0.05, 0.05])

        self.ekf = EKF(robot=robot, P0=P0, Q=Q, R=R)
        self.start_time = time.time()

        self.state_hist: list[StateVector] = [state0]
        init_rr()

    def update_ekf(self, u: Controls, z: Measurement):
        state_prior, cov_prior = self.ekf.predict(
            state=self.state_hist[-1], controls=u, dt=0.01
        )
        state_new = self.ekf.update(state_prior=state_prior, cov_prior=cov_prior, z=z)
        self.state_hist.append(state_new)

    def on_message(self, ws: websocket.WebSocket, message: str):
        msg = json.loads(message)
        if msg["message_type"] == "score":
            print(msg)

        else:
            sensors = msg["sensors"]

            accdata = sensors[0]["data"]
            gyrodata = sensors[1]["data"]
            gpsdata = sensors[2]["data"]

            z = Measurement(
                float(accdata[0]),
                float(accdata[1]),
                float(gyrodata[0]),
                float(gpsdata[0]),
                float(gpsdata[1]),
            )
            print(z)
            u = Controls(*np.array([1.0, 1.0]))
            self.update_ekf(u=u, z=z)

            # elif gpsdata != sensorhistory[-1][3:]:
            #     print("new!")
            #     sensorhistory.append([*accdata, *gyrodata, *gpsdata])

            inputs = {
                "v_left": 0.0,
                "v_right": 0.0,
            }
            update_rr(self.state_hist, t=time.time() - self.start_time)
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
