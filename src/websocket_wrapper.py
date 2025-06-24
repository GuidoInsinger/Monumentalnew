import json
import time

import websocket

from .robot import Robot
from .visualization import init_rr, update_rr


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
        init_rr(self.robot)  # initialize visualizer

    def on_message(self, ws: websocket.WebSocket, message: str) -> None:
        t = time.time() - self.start_time
        msg = json.loads(message)  # convert to dict
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

    def on_error(self, ws: websocket.WebSocket, error: Exception) -> None:
        print(error)

    def on_close(
        self, ws: websocket.WebSocket, close_status_code: int, close_msg: str
    ) -> None:
        print("### closed ###")

    def on_open(self, ws: websocket.WebSocket) -> None:
        self.start_time = time.time()
        inputs = {  # start by sending control to go forward
            "v_left": 2.0,
            "v_right": 2.0,
        }
        ws.send(json.dumps(inputs))
        print("Opened connection")
