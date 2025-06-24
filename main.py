import json
import time

import numpy as np
import websocket

from src.controller import Controller
from src.datatypes import RobotDimensions, StateVector
from src.EKF import EKF
from src.robot import Robot
from src.viz import init_rr, update_rr


class WebsocketWrapper(websocket.WebSocketApp):
    def __init__(self, robot: Robot):
        super().__init__(  # type:ignore
            "ws://91.99.103.188:8765",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        self.robot = robot
        init_rr(self.robot)

    def on_message(self, ws: websocket.WebSocket, message: str):
        t = time.time() - self.start_time
        msg = json.loads(message)
        if msg["message_type"] == "score":
            print(msg)
            self.close()  # type:ignore

        elif msg["message_type"] == "sensors":
            sensormessages = msg["sensors"]

            self.robot.process_sensordata(sensormessages=sensormessages)

            controls = self.robot.controller.compute_controls(
                state=self.robot.state_hist[-1], t=t
            )

            inputs = {
                "v_left": controls.v_l_desired,
                "v_right": controls.v_r_desired,
            }
            update_rr(self.robot, t=t)

            ws.send(json.dumps(inputs))

    def on_error(self, ws: websocket.WebSocket, error: Exception):
        print(error)

    def on_close(self, ws: websocket.WebSocket, close_status_code: int, close_msg: str):
        print("### closed ###")

    def on_open(self, ws: websocket.WebSocket):
        self.start_time = time.time()
        inputs = {
            "v_left": 2.0,
            "v_right": 2.0,
        }
        ws.send(json.dumps(inputs))
        print("Opened connection")


if __name__ == "__main__":
    # websocket.enableTrace(True)
    r_wheel = 0.5
    w_wheel = 0.05

    d_body = 0.125
    w_body = 0.25
    h_body = 0.05

    state0 = StateVector(*np.array([0.0, 0.0, 0.0, 0.0]))

    cov0 = np.diag([1e-4, 1e-4, 1e-5, 1e-4])

    Q = 0.005 * np.diag([3e-3, 3e-3, 4e-3, 4e-3])
    R = np.diag([0.1, 0.1])

    ekf = EKF(cov0=cov0, Q=Q, R=R)
    controller = Controller(k1=2.5, k2=-8.0, k3=-5.0)
    dimensions = RobotDimensions(
        d_body=d_body, w_body=w_body, h_body=h_body, r_wheel=r_wheel, w_wheel=w_wheel
    )
    robot = Robot(dimensions=dimensions, state0=state0, ekf=ekf, controller=controller)

    viz_ekf = WebsocketWrapper(robot=robot)
    viz_ekf.run_forever()  # type: ignore
